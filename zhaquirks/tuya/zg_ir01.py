"""HOBEIAN ZG-IR01 Tuya infrared blaster quirk.

The public Tuya datapoint map is documented by Rob Jones's MIT-licensed
community quirk and the HOBEIAN definition in zigbee-herdsman-converters:
https://github.com/therealdigitalkiwi/zha-zg-ir01
https://github.com/Koenkk/zigbee-herdsman-converters/blob/master/src/devices/hobeian.ts
"""

import base64
from collections import deque
import logging
from typing import Any

import zigpy.types as t
from zigpy.zcl import foundation
from zigpy.zcl.clusters.general import OnOff

from zhaquirks.device import CustomZigpyDevice
from zhaquirks.tuya.builder import TuyaQuirkBuilder
from zhaquirks.tuya.ts1201 import ZosungIRControl, ZosungIRTransmit

_LOGGER = logging.getLogger(__name__)

_MAX_OUTGOING_CHUNK_LENGTH = 0x32
_MAX_QUEUED_SENDS = 8
# Header + maximum timing payload + terminator + transport padding.
_MAX_LEARN_PACKET_LENGTH = 0x1000C


class IRStudyState(t.enum8):
    """Per-slot IR code study state."""

    Study = 0x00
    Registered = 0x01
    Unregistered = 0x02


class _LearnTransfer:
    """Assemble one learned IR packet without accepting gaps or corruption."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard the active transfer."""
        self.sequence: int | None = None
        self.expected_length = 0
        self.data = bytearray()
        self.received = bytearray()

    @property
    def expected_position(self) -> int:
        """Return the first byte offset not received from the device."""
        try:
            return self.received.index(0)
        except ValueError:
            return self.expected_length

    def start(self, sequence: int, expected_length: int) -> str | None:
        """Start a transfer, returning a validation failure when unsafe."""
        self.reset()
        if expected_length <= 0:
            return "the announced length is empty"
        if expected_length > _MAX_LEARN_PACKET_LENGTH:
            return "the announced length exceeds the Broadlink packet limit"

        self.sequence = sequence
        self.expected_length = expected_length
        self.data = bytearray(expected_length)
        self.received = bytearray(expected_length)
        return None

    def append(
        self,
        sequence: int,
        position: int,
        part: bytes,
        checksum: int,
    ) -> str | None:
        """Append a valid transfer part at its announced position."""
        if self.sequence is None:
            return "no learn transfer is active"
        if sequence != self.sequence:
            return "the transfer sequence does not match"
        if not part:
            return "the transfer part is empty"
        if checksum != sum(part) % 0x100:
            return "the transfer part checksum does not match"
        if position < 0 or position + len(part) > self.expected_length:
            return "the transfer part exceeds the announced length"

        for offset, value in enumerate(part, start=position):
            if self.received[offset] and self.data[offset] != value:
                return "the transfer part conflicts with bytes already received"

        self.data[position : position + len(part)] = part
        self.received[position : position + len(part)] = b"\x01" * len(part)
        return None

    def finish(self, sequence: int) -> tuple[bytes | None, str | None]:
        """Return the packet only after every announced byte was assembled."""
        if self.sequence is None:
            return None, "no learn transfer is active"

        completion_sequences = {self.sequence, (self.sequence + 1) % 0x10000}
        if sequence not in completion_sequences:
            return None, "the transfer sequence does not match"
        if self.expected_position < self.expected_length:
            return None, "the transfer ended before every byte arrived"

        return bytes(self.data), None


class ZGIR01Device(CustomZigpyDevice):
    """Device state required by the ZG-IR01 Zosung transport."""

    last_learned_ir_code = t.CharacterString("")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize per-device transfer state."""
        self.seq = 0
        self.ir_msg_to_send: dict[int, str] = {}
        self._ir_send_queue: deque[tuple[tuple[Any, ...], dict[str, Any]]] = deque()
        self._ir_send_active = False
        self._ir_send_end_served = False
        super().__init__(*args, **kwargs)

    def next_seq(self) -> int:
        """Use the fixed sequence expected by this model's transport."""
        self.seq = 0
        return self.seq


class ZGIR01Control(ZosungIRControl):
    """Serialize sends that share the fixed ZG-IR01 transfer sequence."""

    async def command(self, command_id: Any, *args: Any, **kwargs: Any) -> Any:
        """Queue raw sends until the device consumes the active message."""
        if command_id != self.ServerCommandDefs.IRSend.id:
            return await super().command(command_id, *args, **kwargs)

        device = self.endpoint.device
        if len(device._ir_send_queue) >= _MAX_QUEUED_SENDS:
            raise RuntimeError("ZG-IR01 send queue is full")
        device._ir_send_queue.append((tuple(args), dict(kwargs)))
        if device._ir_send_active:
            return None

        device._ir_send_active = True
        send_args, send_kwargs = device._ir_send_queue.popleft()
        await self._dispatch_send(send_args, send_kwargs)
        return None

    def _start_next_send(self) -> None:
        """Start the oldest queued send when no transfer is active."""
        device = self.endpoint.device
        if device._ir_send_active or not device._ir_send_queue:
            return

        device._ir_send_active = True
        args, kwargs = device._ir_send_queue.popleft()
        self.create_catching_task(self._dispatch_send(args, kwargs))

    async def _dispatch_send(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Dispatch one send through the inherited Zosung framing."""
        self.endpoint.device._ir_send_end_served = False
        try:
            await super().command(self.ServerCommandDefs.IRSend.id, *args, **kwargs)
        except Exception:
            self.endpoint.device._ir_send_active = False
            self._start_next_send()
            raise

    def finish_send(self) -> None:
        """Release the active transfer and dispatch the next queued send."""
        self.endpoint.device._ir_send_active = False
        self.endpoint.device._ir_send_end_served = False
        self._start_next_send()


class ZGIR01Transmit(ZosungIRTransmit):
    """Apply the transfer framing and validation required by ZG-IR01."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize per-cluster learned-packet state."""
        super().__init__(*args, **kwargs)
        self._learn_transfer = _LearnTransfer()

    async def command(self, command_id: Any, *args: Any, **kwargs: Any) -> Any:
        """Suppress default responses for coordinator-to-device frames."""
        kwargs.setdefault("disable_default_response", True)
        return await super().command(command_id, *args, **kwargs)

    def handle_cluster_request(
        self,
        hdr: foundation.ZCLHeader,
        args: Any,
        *,
        dst_addressing: t.AddrMode | None = None,
    ) -> None:
        """Handle Zosung frames with ZG-IR01 validation and chunk sizing."""
        command_id = hdr.command_id
        if command_id == self.ServerCommandDefs.receive_ir_frame_00.id:
            self._start_learn_transfer(hdr, args)
            return
        if command_id == self.ServerCommandDefs.receive_ir_frame_02.id:
            self._send_message_part(hdr, args)
            return
        if command_id == self.ServerCommandDefs.receive_ir_frame_03.id:
            self._append_learn_transfer(hdr, args)
            return
        if command_id == self.ServerCommandDefs.receive_ir_frame_05.id:
            self._finish_learn_transfer(hdr, args)
            return

        super().handle_cluster_request(hdr, args, dst_addressing=dst_addressing)
        if (
            command_id == self.ServerCommandDefs.receive_ir_frame_04.id
            and self.endpoint.device._ir_send_end_served
        ):
            self._finish_outgoing_transfer(int(args.seq))

    def _send_default_response(
        self,
        hdr: foundation.ZCLHeader,
        status: foundation.Status = foundation.Status.SUCCESS,
    ) -> None:
        """Send a default response only when the incoming frame requested one."""
        if not hdr.frame_control.disable_default_response:
            self.send_default_rsp(hdr, status=status)

    def _request_learn_part(self, sequence: int, position: int) -> None:
        """Request the next missing learned-code part."""
        self.create_catching_task(
            super().command(
                self.ServerCommandDefs.receive_ir_frame_02.id,
                seq=sequence,
                position=position,
                maxlen=0x38,
                expect_reply=False,
                disable_default_response=True,
            )
        )

    def _start_learn_transfer(self, hdr: foundation.ZCLHeader, args: Any) -> None:
        sequence = int(args.seq)
        expected_length = int(args.length)
        error = self._learn_transfer.start(sequence, expected_length)
        if error is not None:
            _LOGGER.warning("Rejecting ZG-IR01 learn transfer: %s", error)
            self._send_default_response(hdr, foundation.Status.FAILURE)
            return

        self._send_default_response(hdr)
        self.create_catching_task(
            super().command(
                self.ServerCommandDefs.receive_ir_frame_01.id,
                zero=0,
                seq=sequence,
                length=expected_length,
                unk1=args.unk1,
                clusterid=args.clusterid,
                unk2=args.unk2,
                cmd=args.cmd,
                unk3=args.unk3,
                expect_reply=True,
                disable_default_response=True,
            )
        )
        self._request_learn_part(sequence, 0)

    def _send_message_part(self, hdr: foundation.ZCLHeader, args: Any) -> None:
        self._send_default_response(hdr)
        sequence = int(args.seq)
        message = self.endpoint.device.ir_msg_to_send.get(sequence)
        if message is None:
            _LOGGER.warning(
                "Ignoring ZG-IR01 transfer request for unknown sequence %s", sequence
            )
            return

        position = int(args.position)
        max_length = min(int(args.maxlen), _MAX_OUTGOING_CHUNK_LENGTH)
        if position == len(message):
            if self.endpoint.device._ir_send_end_served:
                self._finish_outgoing_transfer(sequence)
            else:
                _LOGGER.warning(
                    "Ignoring premature ZG-IR01 transfer completion (sequence=%s)",
                    sequence,
                )
            return
        if position < 0 or position > len(message) or max_length <= 0:
            _LOGGER.warning(
                "Ignoring invalid ZG-IR01 transfer request "
                "(sequence=%s, position=%s, max_length=%s)",
                sequence,
                position,
                max_length,
            )
            return

        part = message[position : position + max_length].encode()
        if position + len(part) == len(message):
            self.endpoint.device._ir_send_end_served = True
        self.create_catching_task(
            super().command(
                self.ClientCommandDefs.resp_ir_frame_03.id,
                zero=0,
                seq=sequence,
                position=position,
                msgpart=part,
                msgpartcrc=sum(part) % 0x100,
                expect_reply=False,
                disable_default_response=True,
            )
        )

    def _finish_outgoing_transfer(self, sequence: int) -> None:
        """Clear a consumed message and release the next serialized send."""
        message = self.endpoint.device.ir_msg_to_send.pop(sequence, None)
        if message is not None:
            self.endpoint.zosung_ircontrol.finish_send()

    def _append_learn_transfer(self, hdr: foundation.ZCLHeader, args: Any) -> None:
        sequence = int(args.seq)
        part = bytes(args.msgpart)
        error = self._learn_transfer.append(
            sequence,
            int(args.position),
            part,
            int(args.msgpartcrc),
        )
        if error is not None:
            _LOGGER.warning(
                "Rejecting ZG-IR01 learn transfer part: %s "
                "(sequence=%s, position=%s, bytes=%s)",
                error,
                sequence,
                int(args.position),
                len(part),
            )
            self._send_default_response(hdr, foundation.Status.FAILURE)
            if sequence == self._learn_transfer.sequence:
                self._request_learn_part(
                    sequence, self._learn_transfer.expected_position
                )
            return

        self._send_default_response(hdr)
        if (
            self._learn_transfer.expected_position
            < self._learn_transfer.expected_length
        ):
            self._request_learn_part(sequence, self._learn_transfer.expected_position)
            return

        self.create_catching_task(
            super().command(
                self.ServerCommandDefs.receive_ir_frame_04.id,
                zero0=0,
                seq=sequence,
                zero1=0,
                expect_reply=False,
                disable_default_response=True,
            )
        )

    def _finish_learn_transfer(self, hdr: foundation.ZCLHeader, args: Any) -> None:
        sequence = int(args.seq)
        packet, error = self._learn_transfer.finish(sequence)
        if error is not None:
            _LOGGER.warning(
                "Rejecting incomplete ZG-IR01 learn transfer: %s "
                "(sequence=%s, first_missing=%s, announced_length=%s)",
                error,
                sequence,
                self._learn_transfer.expected_position,
                self._learn_transfer.expected_length,
            )
            self._send_default_response(hdr, foundation.Status.FAILURE)
            expected_sequence = self._learn_transfer.sequence
            if expected_sequence is not None and sequence in {
                expected_sequence,
                (expected_sequence + 1) % 0x10000,
            }:
                self._request_learn_part(
                    expected_sequence, self._learn_transfer.expected_position
                )
            return

        assert packet is not None
        self._send_default_response(hdr)
        self.endpoint.device.last_learned_ir_code = base64.b64encode(packet).decode()
        self._learn_transfer.reset()
        self.create_catching_task(
            self.endpoint.zosung_ircontrol.command(
                self.endpoint.zosung_ircontrol.ServerCommandDefs.IRLearn.id,
                on_off=False,
                expect_reply=False,
            )
        )


_builder = (
    TuyaQuirkBuilder("HOBEIAN", "ZG-IR01")
    .zigpy_device_class(ZGIR01Device)
    .replace_cluster_occurrences(
        ZGIR01Control,
        replace_client_instances=False,
    )
    .replace_cluster_occurrences(
        ZGIR01Transmit,
        replace_client_instances=False,
    )
    .prevent_default_entity_creation(endpoint_id=1, cluster_id=OnOff.cluster_id)
)

for channel in range(1, 7):
    _builder.tuya_switch(
        dp_id=channel,
        attribute_name=f"switch_{channel}",
        translation_key=f"switch_{channel}",
        fallback_name=f"Switch {channel}",
    )

for channel in range(1, 7):
    for state, offset in (("on", 0), ("off", 1)):
        _builder.tuya_enum(
            dp_id=118 + channel * 2 + offset,
            attribute_name=f"switch_{channel}_{state}_code",
            enum_class=IRStudyState,
            translation_key=f"switch_{channel}_{state}_code",
            fallback_name=f"Switch {channel} {state} code",
        )

_builder.add_to_registry()
