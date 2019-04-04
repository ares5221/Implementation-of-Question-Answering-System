#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json

from aiohttp import web
import asyncio
from bertClient import getBestAnswer
from bertClientRun2 import getBestAnswer2bySimilyQuestionByq2qModel
from bertClientRun3 import getBestAnswer3bySimilyQuestionByQ2QandQ2AModel
import time, os, csv

path = os.path.abspath('..')
save_log_dir = path + '/log/resultLog.csv'


async def handle(request):
    varDict = request.query
    question = varDict['question']
    print('从url解析得到的查询问题信息：', question)
    # 方法一,比较余弦相似度查找相似问题
    res1, res2 = getBestAnswer(question)
    ss = {'res1': res1, 'res2': res2}
    # data = json.dumps(ss, ensure_ascii=False)
    # # 方法二, 通过MLP比较查找相似问题
    # res = getBestAnswer2bySimilyQuestionByq2qModel(question)
    # 方法三, 通过MLP比较查找相似问题和对应的答案
    # res = getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(question)
    print('本次咨询结束，当前时间为：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # return web.Response(text=res)
    return web.json_response(ss)


async def result(request):
    varDict = request.query
    question = varDict['question']
    similaryQuestion = varDict['simQue']
    similaryValue = varDict['value']
    print('从url解析得到的查询问题信息：', question, similaryQuestion, similaryValue)
    news = [question, similaryQuestion, similaryValue]
    with open(save_log_dir, 'a', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(news)
    print('回传信息已经保存在log文件中!!!!!')
    return web.Response(text='回传信息已经保存OK')


async def init_app():
    app = web.Application()
    # app.router.add_get('/', index)
    app.router.add_get('/', handle)
    app.router.add_get('/result', result)  # 用来处理回传的问题/相似问题/相似度(0,1)
    return app


# 简易服务器搭建，启动服务后可以通过ip浏览器访问 或者 http://127.0.0.1:8080/?question=怎么办
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app())
    web.run_app(app)
