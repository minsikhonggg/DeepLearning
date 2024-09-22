is_simple_core = True # 33단계 부터 False, core.py 사용

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
# else:
#     from dezero.core import Variable


setup_variable() # 연산자 오버로드 -> Variable 사용