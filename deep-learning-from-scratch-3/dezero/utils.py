import os
import subprocess

# 로컬에서만 사용, get_dot_graph 함수 전용
# Variable 인스턴스를 DOT 언어로 변환하는 편의 함수 구현
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)

# DeZero 함수를 DOT 언어로 변환하는 편의 함수 구현
def _dot_func(f):
    dot_func = '{} [lable="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'

    # 함수와 입력 변수의 관계 기술
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    
    # 함수와 출력 변수의 관계 기술 
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # y는 약한 참조. 17.4참고

    return txt


def get_dot_graph(output, verbose=True):
    '''
    - backward 구조와 거의 유사
    - 어떤 노드가 존재하는가 / 어떤 노드끼리 연결되는가
    - generation 사용 X
    '''

    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose) # txt에 ouput 추가

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func) # txt에 func 추가
        for x in func.inputs:
            txt += _dot_var(x, verbose) # txt에 input 추가

            if x.creator is not None:
                add_func(x.creator)
    
    return 'digraph g {\n' + txt + '}'


# 이미지 변환
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1. dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir): # ~/ .dezro 디렉터리가 없다면 새로 생성
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 2. dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:] # 확장자 png, pdf 등
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)