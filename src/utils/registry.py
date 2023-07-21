

class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module):
        module_name = module.__name__
        self[module_name] = module
        return module


# Test program
if __name__ == "__main__":
    # Create an instance of the Registry class
    my_registry = Registry()

    # Define some example modules
    @my_registry.register
    class ModuleA:
        def do(self):
            print('A class')


    class ModuleB:
        pass


    class ModuleC:
        pass


    # Register the modules
    # my_registry.register(ModuleA)
    my_registry.register(ModuleB)
    my_registry.register(ModuleC)

    # Access the registered modules
    print(my_registry["ModuleA"])  # Output: <class '__main__.ModuleA'>
    print(my_registry["ModuleB"])  # Output: <class '__main__.ModuleB'>
    print(my_registry["ModuleC"])  # Output: <class '__main__.ModuleC'>

    A = my_registry["ModuleA"]()
    A.do()
