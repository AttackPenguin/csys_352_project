import multiprocessing as mp

def foo(q):
    q.put('hello')


def run():
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
