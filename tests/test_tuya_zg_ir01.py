"""Tests for the HOBEIAN ZG-IR01 infrared blaster quirk."""

import base64
from unittest import mock

import pytest
from zha.quirks import DEVICE_REGISTRY
import zigpy.types as t
from zigpy.zcl import ClusterType, foundation
from zigpy.zcl.clusters.general import OnOff, PowerConfiguration
from zigpy.zcl.clusters.measurement import RelativeHumidity, TemperatureMeasurement

from tests.common import wait_for_zigpy_tasks
import zhaquirks
from zhaquirks.builder import EntityPlatform
from zhaquirks.tuya import TUYA_CLUSTER_ID, TuyaCommand, TuyaData, TuyaDatapointData
from zhaquirks.tuya.mcu import TuyaMCUCluster
from zhaquirks.tuya.ts1201 import ZosungIRControl
from zhaquirks.tuya.zg_ir01 import (
    IRStudyState,
    ZGIR01Control,
    ZGIR01Device,
    ZGIR01Transmit,
    _LearnTransfer,
)

zhaquirks.setup()

_BASE_CLUSTERS = {
    1: {
        OnOff.cluster_id: ClusterType.Server,
        PowerConfiguration.cluster_id: ClusterType.Server,
        TemperatureMeasurement.cluster_id: ClusterType.Server,
        RelativeHumidity.cluster_id: ClusterType.Server,
        TUYA_CLUSTER_ID: ClusterType.Server,
    }
}


def _cluster_ids(*, raw_transport: bool) -> dict[int, dict[int, ClusterType]]:
    """Return an interviewed ZG-IR01 cluster set."""
    clusters = {1: _BASE_CLUSTERS[1].copy()}
    if raw_transport:
        clusters[1].update(
            {
                ZosungIRControl.cluster_id: ClusterType.Server,
                ZGIR01Transmit.cluster_id: ClusterType.Server,
            }
        )
    return clusters


@pytest.mark.parametrize("raw_transport", (False, True))
def test_zg_ir01_matches_capabilities(zigpy_device_from_v2_quirk, raw_transport):
    """Test the HOBEIAN device with both advertised cluster shapes."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=raw_transport),
    )
    endpoint = device.endpoints[1]

    assert isinstance(device, ZGIR01Device)
    assert device.next_seq() == 0
    assert device.seq == 0
    assert isinstance(endpoint.tuya_manufacturer, TuyaMCUCluster)
    assert type(endpoint.power) is PowerConfiguration
    assert type(endpoint.temperature) is TemperatureMeasurement
    assert type(endpoint.humidity) is RelativeHumidity
    assert type(endpoint.on_off) is OnOff

    if raw_transport:
        assert isinstance(endpoint.zosung_ircontrol, ZGIR01Control)
        assert isinstance(endpoint.zosung_irtransmit, ZGIR01Transmit)
    else:
        assert ZosungIRControl.cluster_id not in endpoint.in_clusters
        assert ZGIR01Transmit.cluster_id not in endpoint.in_clusters

    entry = DEVICE_REGISTRY.match_entry(device)
    definition = entry.zha_device_factory.quirk_definition
    metadata = definition.entity_metadata
    assert len(metadata) == 18
    assert sum(item.entity_platform is EntityPlatform.SWITCH for item in metadata) == 6
    assert sum(item.entity_platform is EntityPlatform.SELECT for item in metadata) == 12
    assert {item.fallback_name for item in metadata} >= {
        "Switch 1",
        "Switch 6",
        "Switch 1 on code",
        "Switch 6 off code",
    }
    assert any(
        item.endpoint_id == 1 and item.cluster_id == OnOff.cluster_id
        for item in definition.disabled_default_entities
    )


async def test_zg_ir01_datapoint_reports_and_writes(zigpy_device_from_v2_quirk):
    """Test representative channel and study-state datapoint round trips."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=False),
    )
    cluster = device.endpoints[1].tuya_manufacturer

    status = cluster.handle_get_data(
        TuyaCommand(
            status=0,
            tsn=1,
            datapoints=[
                TuyaDatapointData(1, TuyaData(t.Bool.true)),
                TuyaDatapointData(121, TuyaData(IRStudyState.Registered)),
            ],
        )
    )
    assert status == foundation.Status.SUCCESS
    assert cluster.get("switch_1") is t.Bool.true
    assert cluster.get("switch_1_off_code") == IRStudyState.Registered

    with mock.patch.object(
        cluster.endpoint, "request", return_value=foundation.Status.SUCCESS
    ) as request_mock:
        await cluster.write_attributes({"switch_1": t.Bool.true})
        await wait_for_zigpy_tasks()
        assert request_mock.call_args.kwargs["data"].endswith(
            b"\x01\x01\x01\x00\x01\x01"
        )

        await cluster.write_attributes({"switch_1_off_code": IRStudyState.Study})
        await wait_for_zigpy_tasks()
        assert request_mock.call_args.kwargs["data"].endswith(b"\x79\x04\x00\x01\x00")


def test_zg_ir01_learn_transfer_validation():
    """Test packet assembly, retries, overlap checks, and completion rules."""
    transfer = _LearnTransfer()
    packet = b"abcdefgh"

    assert transfer.start(7, len(packet)) is None
    assert transfer.append(7, 4, packet[4:], sum(packet[4:]) % 0x100) is None
    assert transfer.expected_position == 0
    assert transfer.append(7, 0, packet[:4], 0) == (
        "the transfer part checksum does not match"
    )
    assert transfer.append(7, 0, packet[:4], sum(packet[:4]) % 0x100) is None
    assert transfer.append(7, 0, packet[:4], sum(packet[:4]) % 0x100) is None
    assert transfer.append(7, 1, b"BAD", sum(b"BAD") % 0x100) == (
        "the transfer part conflicts with bytes already received"
    )
    assert transfer.finish(8) == (packet, None)


def test_zg_ir01_incomplete_and_invalid_learn_transfers():
    """Test unsafe lengths, sequences, bounds, and premature completion."""
    transfer = _LearnTransfer()

    assert transfer.append(1, 0, b"data", sum(b"data") % 0x100) == (
        "no learn transfer is active"
    )
    assert transfer.finish(1) == (None, "no learn transfer is active")
    assert transfer.start(1, 0) == "the announced length is empty"
    assert transfer.start(1, 0x1000C) is None
    assert transfer.start(1, 0x1000D) == (
        "the announced length exceeds the Broadlink packet limit"
    )
    assert transfer.start(1, 4) is None
    assert transfer.append(1, 0, b"", 0) == "the transfer part is empty"
    assert transfer.append(2, 0, b"data", sum(b"data") % 0x100) == (
        "the transfer sequence does not match"
    )
    assert transfer.append(1, 0, b"overflow", sum(b"overflow") % 0x100) == (
        "the transfer part exceeds the announced length"
    )
    assert transfer.append(1, 0, b"da", sum(b"da") % 0x100) is None
    assert transfer.finish(1) == (
        None,
        "the transfer ended before every byte arrived",
    )
    assert transfer.finish(3) == (None, "the transfer sequence does not match")


async def test_zg_ir01_outgoing_transport_caps_parts(
    zigpy_device_from_v2_quirk,
):
    """Test raw transport uses sequence zero and sends at most 50 bytes."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    transmit = device.endpoints[1].zosung_irtransmit
    message = "A" * 80
    device.ir_msg_to_send[0] = message

    with mock.patch.object(
        transmit.endpoint, "request", return_value=foundation.Status.SUCCESS
    ) as request_mock:
        hdr, args = transmit.deserialize(bytes.fromhex("11670200000000000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    request = request_mock.call_args.kwargs
    assert request["cluster"] == ZGIR01Transmit.cluster_id
    assert request["command_id"] == (
        ZGIR01Transmit.ClientCommandDefs.resp_ir_frame_03.id
    )
    sent_hdr, sent_args = transmit.deserialize(request["data"])
    assert sent_hdr.command_id == ZGIR01Transmit.ClientCommandDefs.resp_ir_frame_03.id
    assert sent_args.seq == 0
    assert sent_args.position == 0
    assert bytes(sent_args.msgpart) == b"A" * 50
    assert sent_args.msgpartcrc == sum(b"A" * 50) % 0x100

    with mock.patch.object(transmit.endpoint, "request") as premature_request_mock:
        hdr, args = transmit.deserialize(bytes.fromhex("11670200005000000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        premature_request_mock.assert_not_called()
    assert device.ir_msg_to_send == {0: message}

    with mock.patch.object(
        transmit.endpoint, "request", return_value=foundation.Status.SUCCESS
    ):
        hdr, args = transmit.deserialize(bytes.fromhex("11670200003200000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    with mock.patch.object(transmit.endpoint, "request") as unknown_request_mock:
        hdr, args = transmit.deserialize(bytes.fromhex("11670201000000000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        unknown_request_mock.assert_not_called()

    with mock.patch.object(transmit.endpoint, "request") as completion_request_mock:
        hdr, args = transmit.deserialize(bytes.fromhex("11670200005000000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        completion_request_mock.assert_not_called()
    assert device.ir_msg_to_send == {}

    device.ir_msg_to_send[0] = message
    with mock.patch.object(transmit.endpoint, "request") as invalid_request_mock:
        hdr, args = transmit.deserialize(bytes.fromhex("11670200005100000040"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        invalid_request_mock.assert_not_called()
    assert device.ir_msg_to_send == {0: message}


async def test_zg_ir01_serializes_fixed_sequence_sends(
    zigpy_device_from_v2_quirk,
):
    """Test a second send waits until the device consumes the first."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    control = device.endpoints[1].zosung_ircontrol
    transmit = device.endpoints[1].zosung_irtransmit

    with mock.patch.object(
        control.endpoint, "request", return_value=foundation.Status.SUCCESS
    ):
        await control.command(control.ServerCommandDefs.IRSend.id, code="first")
        first_message = device.ir_msg_to_send[0]
        assert device.seq == 0
        assert '"key_code":"first"' in first_message

        await control.command(control.ServerCommandDefs.IRSend.id, code="second")
        await wait_for_zigpy_tasks()

        assert len(device._ir_send_queue) == 1

        for position in range(0, len(first_message), 50):
            request = (
                bytes.fromhex("1167020000") + position.to_bytes(4, "little") + b"\x40"
            )
            hdr, args = transmit.deserialize(request)
            transmit.handle_message(hdr, args)
            await wait_for_zigpy_tasks()

        terminal = (
            bytes.fromhex("1167020000")
            + len(first_message).to_bytes(4, "little")
            + b"\x40"
        )
        hdr, args = transmit.deserialize(terminal)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

        second_message = device.ir_msg_to_send[0]
        assert '"key_code":"second"' in second_message
        assert not device._ir_send_queue

        duplicate_completion = bytes.fromhex("0169040000000000")
        hdr, args = transmit.deserialize(duplicate_completion)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        assert device.ir_msg_to_send == {0: second_message}

        for position in range(0, len(second_message), 50):
            request = (
                bytes.fromhex("1167020000") + position.to_bytes(4, "little") + b"\x40"
            )
            hdr, args = transmit.deserialize(request)
            transmit.handle_message(hdr, args)
            await wait_for_zigpy_tasks()

        terminal = (
            bytes.fromhex("1167020000")
            + len(second_message).to_bytes(4, "little")
            + b"\x40"
        )
        hdr, args = transmit.deserialize(terminal)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    assert device.ir_msg_to_send == {}
    assert device._ir_send_active is False


async def test_zg_ir01_recovers_from_send_dispatch_failure(
    zigpy_device_from_v2_quirk,
):
    """Test a failed dispatch releases the fixed-sequence send queue."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    control = device.endpoints[1].zosung_ircontrol

    with (
        mock.patch(
            "zhaquirks.tuya.ts1201.ZosungIRControl.command",
            new=mock.AsyncMock(side_effect=RuntimeError("synthetic failure")),
        ),
        pytest.raises(RuntimeError, match="synthetic failure"),
    ):
        await control.command(control.ServerCommandDefs.IRSend.id, code="first")

    assert device._ir_send_active is False


async def test_zg_ir01_bounds_the_send_queue(zigpy_device_from_v2_quirk):
    """Test a stalled device cannot accumulate an unbounded send queue."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    control = device.endpoints[1].zosung_ircontrol
    device._ir_send_active = True

    for index in range(8):
        await control.command(control.ServerCommandDefs.IRSend.id, code=str(index))

    with pytest.raises(RuntimeError, match="send queue is full"):
        await control.command(control.ServerCommandDefs.IRSend.id, code="overflow")


async def test_zg_ir01_transport_command_defaults_and_completion(
    zigpy_device_from_v2_quirk,
):
    """Test outgoing defaults are suppressed and completed sends are cleared."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    transmit = device.endpoints[1].zosung_irtransmit

    with mock.patch(
        "zhaquirks.tuya.ts1201.ZosungIRTransmit.command",
        new=mock.AsyncMock(return_value=foundation.Status.SUCCESS),
    ) as command_mock:
        await transmit.command(
            ZGIR01Transmit.ClientCommandDefs.resp_ir_frame_05.id,
            seq=0,
            zero=0,
        )
        command_mock.assert_awaited_once_with(
            ZGIR01Transmit.ClientCommandDefs.resp_ir_frame_05.id,
            seq=0,
            zero=0,
            disable_default_response=True,
        )

    device.ir_msg_to_send[0] = "synthetic"
    device._ir_send_end_served = True
    with mock.patch.object(
        transmit.endpoint, "request", return_value=foundation.Status.SUCCESS
    ):
        hdr, args = transmit.deserialize(bytes.fromhex("0169040000000000"))
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    assert device.ir_msg_to_send == {}


async def test_zg_ir01_rejects_invalid_learn_start(zigpy_device_from_v2_quirk):
    """Test a zero-length learn announcement is rejected before allocation."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    transmit = device.endpoints[1].zosung_irtransmit
    start = bytes.fromhex("05021001000700000000000000000004e001020000")

    with (
        mock.patch.object(transmit, "send_default_rsp") as default_response_mock,
        mock.patch.object(transmit.endpoint, "request") as request_mock,
    ):
        hdr, args = transmit.deserialize(start)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    default_response_mock.assert_called_once_with(
        hdr,
        status=foundation.Status.FAILURE,
    )
    request_mock.assert_not_called()


async def test_zg_ir01_learn_frames_publish_only_complete_packets(
    zigpy_device_from_v2_quirk,
):
    """Test real ZCL learn frames reject gaps and publish an exact packet."""
    device = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
    )
    transmit = device.endpoints[1].zosung_irtransmit
    packet = b"synthetic-ir"
    sequence = 7

    with mock.patch.object(
        transmit.endpoint, "request", return_value=foundation.Status.SUCCESS
    ):
        start = (
            bytes.fromhex("0502100100")
            + sequence.to_bytes(2, "little")
            + len(packet).to_bytes(4, "little")
            + bytes.fromhex("0000000004e001020000")
        )
        hdr, args = transmit.deserialize(start)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

        premature = bytes.fromhex("09670508000000")
        hdr, args = transmit.deserialize(premature)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        assert device.last_learned_ir_code == ""

        first_part = packet[:5]
        bad_part = (
            bytes.fromhex("09680300")
            + sequence.to_bytes(2, "little")
            + (0).to_bytes(4, "little")
            + bytes((len(first_part),))
            + first_part
            + b"\x00"
        )
        hdr, args = transmit.deserialize(bad_part)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()
        assert device.last_learned_ir_code == ""

        first_part_frame = bad_part[:-1] + bytes((sum(first_part) % 0x100,))
        hdr, args = transmit.deserialize(first_part_frame)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

        remaining = packet[len(first_part) :]
        final_part = (
            bytes.fromhex("09690300")
            + sequence.to_bytes(2, "little")
            + len(first_part).to_bytes(4, "little")
            + bytes((len(remaining),))
            + remaining
            + bytes((sum(remaining) % 0x100,))
        )
        hdr, args = transmit.deserialize(final_part)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

        complete = bytes.fromhex("09690508000000")
        hdr, args = transmit.deserialize(complete)
        transmit.handle_message(hdr, args)
        await wait_for_zigpy_tasks()

    assert device.last_learned_ir_code == base64.b64encode(packet).decode()


async def test_zg_ir01_transport_state_is_per_instance(
    zigpy_device_from_v2_quirk,
):
    """Test learned and outgoing transfer state is not shared by devices."""
    first = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
        ieee=t.EUI64(b"ZgIrDev1"),
    )
    second = zigpy_device_from_v2_quirk(
        "HOBEIAN",
        "ZG-IR01",
        cluster_ids=_cluster_ids(raw_transport=True),
        ieee=t.EUI64(b"ZgIrDev2"),
    )

    first.ir_msg_to_send[0] = "synthetic"
    first.endpoints[1].zosung_irtransmit._learn_transfer.start(1, 4)

    assert second.ir_msg_to_send == {}
    assert second.endpoints[1].zosung_irtransmit._learn_transfer.sequence is None
