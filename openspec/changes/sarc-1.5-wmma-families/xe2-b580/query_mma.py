#!/usr/bin/env python3
"""Read cooperative-matrix properties from Vulkan device zero; submit no work."""
import ctypes as c
import json


class Application(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("pApplicationName", c.c_char_p), ("applicationVersion", c.c_uint32),
                ("pEngineName", c.c_char_p), ("engineVersion", c.c_uint32),
                ("apiVersion", c.c_uint32)]


class InstanceInfo(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("flags", c.c_uint32), ("pApplicationInfo", c.POINTER(Application)),
                ("enabledLayerCount", c.c_uint32), ("ppEnabledLayerNames", c.c_void_p),
                ("enabledExtensionCount", c.c_uint32), ("ppEnabledExtensionNames", c.c_void_p)]


class Matrix(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p)] + [
        (name, c.c_uint32) for name in
        ("MSize", "NSize", "KSize", "AType", "BType", "CType", "ResultType",
         "saturatingAccumulation", "scope")]


vk = c.CDLL("libvulkan.so.1")
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
vk.vkGetInstanceProcAddr.argtypes = [c.c_void_p, c.c_char_p]
vk.vkGetInstanceProcAddr.restype = c.c_void_p
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
app = Application(sType=0, apiVersion=(1 << 22) | (1 << 12))
info = InstanceInfo(sType=1, pApplicationInfo=c.pointer(app))
instance = c.c_void_p()
assert vk.vkCreateInstance(c.byref(info), None, c.byref(instance)) == 0
try:
    count = c.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) == 0
    devices = (c.c_void_p * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) == 0
    address = vk.vkGetInstanceProcAddr(instance, b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
    assert address, "cooperative-matrix query unavailable"
    query = c.CFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(Matrix))(address)
    count = c.c_uint32()
    assert query(devices[0], c.byref(count), None) == 0
    matrices = (Matrix * count.value)()
    for matrix in matrices:
        matrix.sType = 1000506001
    assert query(devices[0], c.byref(count), matrices) == 0
    types = {0: "float16", 1: "float32", 2: "float64", 3: "sint8", 4: "sint16",
             5: "sint32", 6: "sint64", 7: "uint8", 8: "uint16", 9: "uint32", 10: "uint64", 1000141000: "bfloat16"}
    rows = []
    for matrix in matrices[:count.value]:
        row = {name: getattr(matrix, name) for name, _ in Matrix._fields_[2:]}
        for name in ("AType", "BType", "CType", "ResultType"):
            row[name] = types.get(row[name], row[name])
        rows.append(row)
    print(json.dumps(rows, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
