# test_skilltree_dialog.py
from skilltree import SkillTree, search_and_show
import time

def run_demo():
    st = SkillTree()
    st.initialize("root knowledge")

    # 多轮对话（示例），把“用户话语”写入树
    turns = [
        ("user: 我想学习 C++ 的类和继承", "knowledge"),
        ("assistant: 好的，你先学会类的基本语法：class A { public: int x; };", "knowledge"),
        ("user: 如何实现虚函数和多态？", "question"),
        ("assistant: 在基类用 virtual 修饰函数，子类 override，实现多态", "knowledge"),
        ("user: 我想写一个虚析构函数的例子", "question")
    ]

    # 把带标签的知识写入 tree（仅把 knowledge 插入）
    for text, tag in turns:
        if tag == "knowledge":
            st.add(text, timestamp=time.time())
            print("加入知识:", text)

    # 现在做检索：模拟用户问“怎么实现多态？”
    query = "怎么实现 C++ 的多态？"
    print("\n检索查询：", query)
    res = search_and_show(st, query, top_k=3)
    for r in res:
        print("score:", round(r['score'], 4))
        print("node:", r['node_text'])
        print("path:", " -> ".join(r['path_texts']))
        print("children_count:", r['children_count'])
        print("----")

    # 展示树序列化（简短）
    print("\n当前树（简短序列化）")
    print(st.serialize()[:1200])

if __name__ == "__main__":
    run_demo()
