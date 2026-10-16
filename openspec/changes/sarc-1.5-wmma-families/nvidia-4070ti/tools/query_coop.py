import ctypes as c
import json

class Application(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('pApplicationName', c.c_char_p), ('applicationVersion', c.c_uint32), ('pEngineName', c.c_char_p), ('engineVersion', c.c_uint32), ('apiVersion', c.c_uint32)]
class InstanceCreate(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32), ('pApplicationInfo', c.POINTER(Application)), ('enabledLayerCount', c.c_uint32), ('ppEnabledLayerNames', c.c_void_p), ('enabledExtensionCount', c.c_uint32), ('ppEnabledExtensionNames', c.c_void_p)]
class Matrix(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p)] + [(n, c.c_uint32) for n in ['MSize', 'NSize', 'KSize', 'AType', 'BType', 'CType', 'ResultType', 'saturatingAccumulation', 'scope']]

vk = c.CDLL('libvulkan.so.1')
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceCreate), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
vk.vkGetInstanceProcAddr.argtypes = [c.c_void_p, c.c_char_p]
vk.vkGetInstanceProcAddr.restype = c.c_void_p
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
app = Application(sType=0, pApplicationName=b'wmma-preflight', apiVersion=(1 << 22) | (3 << 12))
info = InstanceCreate(sType=1, pApplicationInfo=c.pointer(app))
instance = c.c_void_p()
assert vk.vkCreateInstance(c.byref(info), None, c.byref(instance)) == 0
try:
    count = c.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) == 0
    devices = (c.c_void_p * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) == 0
    address = vk.vkGetInstanceProcAddr(instance, b'vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR')
    assert address
    query = c.CFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(Matrix))(address)
    result = []
    for index, device in enumerate(devices):
        properties = c.create_string_buffer(4096)
        vk.vkGetPhysicalDeviceProperties(device, properties)
        vendor = int.from_bytes(properties.raw[8:12], 'little')
        name = properties.raw[20:276].split(b'\0')[0].decode()
        if vendor != 0x10de:
            continue
        assert index == 0 and name == 'NVIDIA GeForce RTX 4070 Ti SUPER'
        n = c.c_uint32()
        assert query(device, c.byref(n), None) == 0
        matrices = (Matrix * n.value)()
        for matrix in matrices:
            matrix.sType = 1000506001
        assert query(device, c.byref(n), matrices) == 0
        result.append({'device_index': index, 'vendor_id': vendor, 'name': name, 'matrices': [{k: getattr(matrix, k) for k, _ in Matrix._fields_[2:]} for matrix in matrices[:n.value]]})
    assert len(result) == 1
    print(json.dumps(result, indent=2))
finally:
    vk.vkDestroyInstance(instance, None)
