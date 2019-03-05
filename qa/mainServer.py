#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from aiohttp import web
import asyncio
from bertClient import getBestAnswer


async def handle(request):
    varDict = request.query
    question = varDict['question']
    print('从url解析得到的查询问题信息：', question)
    res = getBestAnswer(question)
    print(res)
    return web.Response(text=res)


async def init_app():
    app = web.Application()
    # app.router.add_route('POST', '/predict', handler.predict, name='predict')
    # app.router.add_get('/', index)
    app.router.add_get('/', handle)
    return app

# Start position
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app())
    web.run_app(app)
