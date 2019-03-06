#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bertClient import getBestAnswer


def testBERTEncode():
    bc = BertClient()
    # print(bc.encode(['我 喜欢 你们', '我 喜 欢 你 们', '我喜欢你们']))  # 模型不需要分词，发现这几种情况得到的编码向量是一样的
    s1 = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没...'
    s2 = '学生上课注意力不集中怎么办？'
    bcs1 = bc.encode([s1])
    bcs2 = bc.encode([s2])
    print(bcs1)
    print('---------test encode----------')
    print(bcs2)
    print(len(bcs1[0]), 'and ', len(bcs2[0]))

if __name__ == '__main__':
    testQ = '怎么样让学生在课堂合作学习中发挥最佳效果？'
    print(getBestAnswer(testQ))
    # testBERTEncode()  #test
