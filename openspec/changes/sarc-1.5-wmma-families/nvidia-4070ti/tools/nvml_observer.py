"""Read-only NVML sampling; process ownership follows the observer's ancestry."""

import ctypes as c
from pathlib import Path
import time


class ProcessInfo(c.Structure):
    _fields_ = [("pid", c.c_uint32), ("usedGpuMemory", c.c_uint64),
                ("gpuInstanceId", c.c_uint32), ("computeInstanceId", c.c_uint32)]


class Utilization(c.Structure):
    _fields_ = [("gpu", c.c_uint32), ("memory", c.c_uint32)]


class Observer:
    def __init__(self, index, owner_pid):
        self.lib = c.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [c.c_uint32, c.POINTER(c.c_void_p)]
        self.lib.nvmlDeviceGetClockInfo.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32)]
        self.lib.nvmlDeviceGetTemperature.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32)]
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [c.c_void_p, c.POINTER(c.c_uint32)]
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [c.c_void_p, c.POINTER(Utilization)]
        self.lib.nvmlDeviceGetCurrentClocksEventReasons.argtypes = [c.c_void_p, c.POINTER(c.c_uint64)]
        self.lib.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(ProcessInfo)]
        self.lib.nvmlDeviceGetGraphicsRunningProcesses_v3.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(ProcessInfo)]
        self.lib.nvmlDeviceGetUUID.argtypes = [c.c_void_p, c.c_char_p, c.c_uint32]
        self.lib.nvmlSystemGetDriverVersion.argtypes = [c.c_char_p, c.c_uint32]
        self.check(self.lib.nvmlInit_v2())
        self.handle = c.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, c.byref(self.handle)))
        self.owner_pid = owner_pid
        stat = Path(f"/proc/{owner_pid}/stat").read_text().rsplit(")", 1)[1].split()
        self.known_own = {owner_pid: stat[19]}
        uuid, driver = c.create_string_buffer(128), c.create_string_buffer(128)
        self.check(self.lib.nvmlDeviceGetUUID(self.handle, uuid, len(uuid)))
        self.check(self.lib.nvmlSystemGetDriverVersion(driver, len(driver)))
        self.identity = {"device_index": index, "unit": uuid.value.decode(), "driver": driver.value.decode()}

    @staticmethod
    def check(result):
        if result != 0:
            raise RuntimeError(f"NVML returned {result}; no retry")

    def own(self, pid):
        path = []
        while pid > 1:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                return True if pid in self.known_own else None
            if self.known_own.get(pid) == stat[19]:
                self.known_own.update(path)
                return True
            path.append((pid, stat[19]))
            pid = int(stat[1])
        return False

    def sample(self):
        start = time.monotonic_ns()
        clock, temperature, power = c.c_uint32(), c.c_uint32(), c.c_uint32()
        reasons, utilization = c.c_uint64(), Utilization()
        self.check(self.lib.nvmlDeviceGetClockInfo(self.handle, 0, c.byref(clock)))
        self.check(self.lib.nvmlDeviceGetTemperature(self.handle, 0, c.byref(temperature)))
        self.check(self.lib.nvmlDeviceGetPowerUsage(self.handle, c.byref(power)))
        self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, c.byref(utilization)))
        self.check(self.lib.nvmlDeviceGetCurrentClocksEventReasons(self.handle, c.byref(reasons)))
        clients = []
        for kind, query in [("compute", self.lib.nvmlDeviceGetComputeRunningProcesses_v3),
                            ("graphics", self.lib.nvmlDeviceGetGraphicsRunningProcesses_v3)]:
            count = c.c_uint32(1024)
            processes = (ProcessInfo * count.value)()
            self.check(query(self.handle, c.byref(count), processes))
            for process in processes[:count.value]:
                clients.append({"pid": process.pid, "kind": kind, "own": self.own(process.pid),
                                "gpu_memory_bytes": process.usedGpuMemory})
        return {"epoch_us": time.time_ns() // 1000, "monotonic_ns": time.monotonic_ns(),
                "query_us": (time.monotonic_ns() - start) // 1000,
                "clock_mhz": clock.value, "temperature_c": temperature.value,
                "power_w": power.value / 1000, "gpu_utilization": utilization.gpu,
                "event_reasons": reasons.value, "thermal": bool(reasons.value & 0x60),
                "hw_slowdown": bool(reasons.value & 8), "clients": clients}

    @staticmethod
    def unexpected(clients, idle):
        """Clients that are neither ours nor a pre-approved idle foreign client with unchanged memory."""
        return [c for c in clients if c["own"] is not True and idle.get(str(c["pid"])) != c["gpu_memory_bytes"]]

    def close(self):
        self.check(self.lib.nvmlShutdown())
