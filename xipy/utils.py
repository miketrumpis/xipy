

def with_attribute(a):
    def dec(f):
        def runner(obj, *args, **kwargs):
            if not getattr(obj, a, False):
                return
            return f(obj, *args, **kwargs)
        # copy f's info to runner
        for attr in ['func_doc', 'func_name']:
            setattr(runner, attr, getattr(f, attr))
        return runner
    return dec

