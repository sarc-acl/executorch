import ctypes as c
import json


class Application(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("name", c.c_char_p), ("version", c.c_uint32),
                ("engine", c.c_char_p), ("engineVersion", c.c_uint32),
                ("apiVersion", c.c_uint32)]


class InstanceCreate(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("flags", c.c_uint32), ("app", c.POINTER(Application)),
                ("layerCount", c.c_uint32), ("layers", c.c_void_p),
                ("extensionCount", c.c_uint32), ("extensions", c.c_void_p)]


class FormatProperties(c.Structure):
    _fields_ = [("linear", c.c_uint32), ("optimal", c.c_uint32),
                ("buffer", c.c_uint32)]


class ImageProperties(c.Structure):
    _fields_ = [("extent", c.c_uint32 * 3), ("mipLevels", c.c_uint32),
                ("arrayLayers", c.c_uint32), ("sampleCounts", c.c_uint32),
                ("resourceSize", c.c_uint64)]


vk = c.CDLL("libvulkan.so.1")
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceCreate), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
vk.vkGetPhysicalDeviceFormatProperties.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(FormatProperties)]
vk.vkGetPhysicalDeviceImageFormatProperties.argtypes = [c.c_void_p] + [c.c_uint32] * 5 + [c.POINTER(ImageProperties)]
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
app = Application(sType=0, name=b"wmma-format-query", apiVersion=(1 << 22) | (3 << 12))
info = InstanceCreate(sType=1, app=c.pointer(app))
instance = c.c_void_p()
assert vk.vkCreateInstance(c.byref(info), None, c.byref(instance)) == 0
try:
    count = c.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) == 0
    devices = (c.c_void_p * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) == 0
    props = c.create_string_buffer(4096)
    vk.vkGetPhysicalDeviceProperties(devices[0], props)
    assert int.from_bytes(props.raw[8:12], "little") == 0x10DE
    rows = []
    for name, fmt in [("R16G16B16A16_SFLOAT", 97), ("R8G8B8A8_SINT", 42), ("R32G32B32A32_SINT", 108)]:
        features = FormatProperties()
        vk.vkGetPhysicalDeviceFormatProperties(devices[0], fmt, c.byref(features))
        assert features.optimal & 3 == 3, (name, features.optimal)
        for image_type in [1, 2]:
            image = ImageProperties()
            # Optimal 2D/3D images with transfer-src/dst, sampled and storage usage.
            rc = vk.vkGetPhysicalDeviceImageFormatProperties(devices[0], fmt, image_type, 0, 15, 0, c.byref(image))
            assert rc == 0, (name, image_type, rc)
            rows.append({"format": name, "format_enum": fmt, "image_type": image_type,
                         "usage": 15, "optimal_features": features.optimal,
                         "max_extent": list(image.extent), "max_resource_bytes": image.resourceSize})
    print(json.dumps({"device_index": 0, "formats": rows}, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
