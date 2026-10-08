#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
int main() {
  void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) { std::cerr << dlerror() << '\n'; return 1; }
  auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib,"vkGetInstanceProcAddr"));
  if (!get) return 2;
  auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr,"vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo=&app;
  VkInstance instance;
  if (!create || create(&ci,nullptr,&instance)!=VK_SUCCESS) return 3;
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get(instance,"vkEnumeratePhysicalDevices"));
  auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get(instance,"vkGetPhysicalDeviceProperties2"));
  uint32_t count=0;
  if (enumerate(instance,&count,nullptr)!=VK_SUCCESS || !count) return 4;
  std::vector<VkPhysicalDevice> devices(count);
  if (enumerate(instance,&count,devices.data())!=VK_SUCCESS) return 5;
  for (auto device:devices) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p.pNext=&id;
    props(device,&p); std::cout << p.properties.deviceName << " UUID=";
    const char* hex="0123456789abcdef";
    for (auto b:id.deviceUUID) std::cout << hex[b>>4] << hex[b&15];
    std::cout << '\n';
  }
  reinterpret_cast<PFN_vkDestroyInstance>(get(instance,"vkDestroyInstance"))(instance,nullptr);
  dlclose(lib);
}
