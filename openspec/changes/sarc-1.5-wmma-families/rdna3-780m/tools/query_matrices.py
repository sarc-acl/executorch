import ctypes as c
import json


class Application(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("pApplicationName", c.c_char_p), ("applicationVersion", c.c_uint32),
                ("pEngineName", c.c_char_p), ("engineVersion", c.c_uint32),
                ("apiVersion", c.c_uint32)]


class InstanceInfo(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                ("pApplicationInfo", c.POINTER(Application)),
                ("enabledLayerCount", c.c_uint32), ("ppEnabledLayerNames", c.c_void_p),
                ("enabledExtensionCount", c.c_uint32), ("ppEnabledExtensionNames", c.c_void_p)]


class Matrix(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p)] + [
        (name, c.c_uint32) for name in ("MSize", "NSize", "KSize", "AType", "BType",
                                      "CType", "ResultType", "saturatingAccumulation", "scope")]


vk = c.CDLL("libvulkan.so.1")
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
vk.vkGetInstanceProcAddr.argtypes = [c.c_void_p, c.c_char_p]
vk.vkGetInstanceProcAddr.restype = c.c_void_p
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
app = Application(sType=0, pApplicationName=b"wmma-property-query", apiVersion=(1 << 22) | (3 << 12))
info = InstanceInfo(sType=1, pApplicationInfo=c.pointer(app))
instance = c.c_void_p()
rc = vk.vkCreateInstance(c.byref(info), None, c.byref(instance))
if rc != 0:
    raise RuntimeError(f"vkCreateInstance: {rc}")
try:
    count = c.c_uint32()
    rc = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
    if rc != 0 or count.value == 0:
        raise RuntimeError(f"vkEnumeratePhysicalDevices: {rc}, count={count.value}")
    devices = (c.c_void_p * count.value)()
    rc = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices)
    if rc != 0:
        raise RuntimeError(f"vkEnumeratePhysicalDevices: {rc}")
    address = vk.vkGetInstanceProcAddr(instance, b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
    if not address:
        raise RuntimeError("KHR matrix query unavailable")
    query = c.CFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(Matrix))(address)
    n = c.c_uint32()
    rc = query(devices[0], c.byref(n), None)
    if rc != 0 or n.value == 0:
        raise RuntimeError(f"matrix count query: {rc}, count={n.value}")
    matrices = (Matrix * n.value)(*[Matrix(sType=1000506001) for _ in range(n.value)])
    rc = query(devices[0], c.byref(n), matrices)
    if rc != 0:
        raise RuntimeError(f"matrix properties query: {rc}")
    print(json.dumps({"device_index": 0, "matrices": [
        {name: getattr(m, name) for name, _ in Matrix._fields_[2:]} for m in matrices[:n.value]
    ]}, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
