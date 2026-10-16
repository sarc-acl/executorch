"""Query supported memory-model features without creating a logical device."""

import ctypes as c
import json
import sys


class Application(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("pApplicationName", c.c_char_p), ("applicationVersion", c.c_uint32),
                ("pEngineName", c.c_char_p), ("engineVersion", c.c_uint32),
                ("apiVersion", c.c_uint32)]


class InstanceCreate(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                ("pApplicationInfo", c.POINTER(Application)),
                ("enabledLayerCount", c.c_uint32), ("ppEnabledLayerNames", c.c_void_p),
                ("enabledExtensionCount", c.c_uint32), ("ppEnabledExtensionNames", c.c_void_p)]


class MemoryModel(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p)] + [
        (name, c.c_uint32) for name in ["vulkanMemoryModel", "vulkanMemoryModelDeviceScope",
                                      "vulkanMemoryModelAvailabilityVisibilityChains"]]


class Features(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("features", c.c_uint32 * 55)]


vk = c.CDLL("libvulkan.so.1")
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceCreate), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
vk.vkGetPhysicalDeviceFeatures2.argtypes = [c.c_void_p, c.POINTER(Features)]
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
application = Application(sType=0, pApplicationName=b"wmma-memory-model-query",
                          apiVersion=(1 << 22) | (3 << 12))
info = InstanceCreate(sType=1, pApplicationInfo=c.pointer(application))
instance = c.c_void_p()
assert vk.vkCreateInstance(c.byref(info), None, c.byref(instance)) == 0
try:
    count = c.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) == 0
    devices = (c.c_void_p * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) == 0
    index = int(sys.argv[1])
    device = devices[index]
    properties = c.create_string_buffer(4096)
    vk.vkGetPhysicalDeviceProperties(device, properties)
    assert int.from_bytes(properties.raw[8:12], "little") == 0x10de
    memory = MemoryModel(sType=1000211000)
    features = Features(sType=1000059000, pNext=c.addressof(memory))
    vk.vkGetPhysicalDeviceFeatures2(device, c.byref(features))
    print(json.dumps({"device_index": index,
                      "name": properties.raw[20:276].split(b"\0")[0].decode(),
                      "supported": {name: bool(getattr(memory, name)) for name, _ in MemoryModel._fields_[2:]},
                      "scope": "physical-device support only; no logical device created or feature enabled"}, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
