#!/usr/bin/env python

# Simple cipher demonstrating the power of the one-time pad

n2l = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
l2n = {n2l[n]: n for n in range(len(n2l))}
Q = len(n2l)

sensim = """WE WILL ATTACK AT 1200"""
clearm = """I WANT APPLE ICE CREAM"""
wrongm = """WE WILL ATTACK AT 0500"""
spooky = """SPOOKY DISTANT ACTIONS"""


def mix(k, x):
    text = []
    m = min(len(k), len(x))
    for i in range(m):
        n = l2n[x[i]]
        np = (n + l2n[k[i]]) % Q
        text.append(n2l[np])
    return "".join(text)


def unmix(k, x):
    text = []
    m = min(len(k), len(x))
    for i in range(m):
        n = l2n[x[i]]
        np = (n + Q - l2n[k[i]]) % Q
        text.append(n2l[np])
    return "".join(text)


def modmult(c, x):
    text = []
    for i in range(len(x)):
        n = l2n[x[i]]
        np = (n * c) % Q
        text.append(n2l[np])
    return "".join(text)


def rot(r, x):
    rotl(n2l[r], x)


def unrot(r, x):
    unrotl(n2l[r], x)


def rotl(rl, x):
    return mix(rl * len(x), x)


def unrotl(rl, x):
    return unmix(rl * len(x), x)


