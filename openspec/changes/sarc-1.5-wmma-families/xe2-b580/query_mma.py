#!/usr/bin/env python3
"""Read cooperative-matrix properties from Vulkan device zero; submit no work."""
import ctypes as c
import json
import sys


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
    if "--formats" in sys.argv:
        class FormatProperties(c.Structure):
            _fields_ = [("linearTilingFeatures", c.c_uint32),
                        ("optimalTilingFeatures", c.c_uint32), ("bufferFeatures", c.c_uint32)]

        class ImageProperties(c.Structure):
            _fields_ = [("maxExtent", c.c_uint32 * 3), ("maxMipLevels", c.c_uint32),
                        ("maxArrayLayers", c.c_uint32), ("sampleCounts", c.c_uint32),
                        ("maxResourceSize", c.c_uint64)]

        vk.vkGetPhysicalDeviceFormatProperties.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(FormatProperties)]
        vk.vkGetPhysicalDeviceImageFormatProperties.argtypes = [
            c.c_void_p, c.c_uint32, c.c_uint32, c.c_uint32, c.c_uint32, c.c_uint32,
            c.POINTER(ImageProperties)]
        formats = []
        for name, code in [("rgba16f", 97), ("rgba32i", 108), ("rgba8i", 42), ("rgba32f", 109)]:
            properties = FormatProperties()
            vk.vkGetPhysicalDeviceFormatProperties(devices[0], code, c.byref(properties))
            assert properties.optimalTilingFeatures & 1, f"{name}: sampled image unsupported"
            assert properties.optimalTilingFeatures & 2, f"{name}: storage image unsupported"
            for image_type in (1, 2):
                for tiling in (0, 1):
                    image = ImageProperties()
                    rc = vk.vkGetPhysicalDeviceImageFormatProperties(devices[0], code, image_type, tiling, 15, 0, c.byref(image))
                    if tiling == 0:
                        assert rc == 0, (name, image_type, rc)
                    formats.append({"format": name, "code": code, "image_type": image_type,
                                    "tiling": tiling, "usage": 15, "rc": rc,
                                    "optimalTilingFeatures": properties.optimalTilingFeatures,
                                    "maxExtent": list(image.maxExtent), "maxResourceSize": image.maxResourceSize})
        probe = ImageProperties()
        probe_rc = vk.vkGetPhysicalDeviceImageFormatProperties(devices[0], 109, 2, 1, 12, 0, c.byref(probe))
        formats.append({"format": "rgba32f", "image_type": 2, "tiling": 1, "usage": 12,
                        "runtime_linear_tiling_probe_rc": probe_rc, "maxExtent": list(probe.maxExtent)})
        print(json.dumps({"matrices": rows, "formats": formats}, indent=2))
    else:
        print(json.dumps(rows, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
