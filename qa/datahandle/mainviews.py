#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from aiohttp import web
from bertClient import getBestAnswer


# 没有用的code
async def index(request):
    varDict = request.query
    question = varDict['gid']
    print(varDict, '@@@@@@@@@@@@@@@@@@@@@@', question)
    res = getBestAnswer(question)
    print(res)
    return web.Response(text=res)
