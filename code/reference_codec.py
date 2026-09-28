#!/usr/bin/env python3
"""Independent, narrow HF01 + SK04/Bit Match reference; NOT a PSO port or Step20 rerun.

This file is standalone for inspecting the storage/extraction format described by
Step30 MATLAB code. Selection uses explicit deterministic groups supplied by caller.
"""
import heapq
import math
import struct
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HF_HDR = struct.Struct('<4sIH')
SK_HDR = struct.Struct('<4sBHHBHIIIIII')
assert SK_HDR.size == 36


def _tree(freq):
    heap = []
    serial = 0
    for s, weight in sorted(freq.items()):
        heapq.heappush(heap, (weight, s, serial, s))
        serial += 1
    if not heap:
        return None
    if len(heap) == 1:
        return heap[0][3]
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        node = (a[3], b[3])
        heapq.heappush(heap, (a[0]+b[0], min(a[1], b[1]), serial, node))
        serial += 1
    return heap[0][3]


def _codes(root):
    out = {}
    def walk(n, path):
        if isinstance(n, int):
            out[n] = path or (0,)
            return
        walk(n[0], path+(0,))
        walk(n[1], path+(1,))
    if root is not None:
        walk(root, ())
    return out


def pack_msb(bits):
    out = bytearray((len(bits)+7)//8)
    for k,b in enumerate(bits):
        if b:
            out[k//8] |= 1 << (7 - k%8)
    return bytes(out)


def pack_lsb(bits):
    out = bytearray((len(bits)+7)//8)
    for k,b in enumerate(bits):
        if b:
            out[k//8] |= 1 << (k%8)
    return bytes(out)


def unpack_lsb(buf, nbits=None):
    nbits = 8*len(buf) if nbits is None else nbits
    if nbits > 8*len(buf):
        raise ValueError('Truncated LSB bits')
    return np.fromiter(((buf[k//8] >> (k%8)) & 1 for k in range(nbits)), dtype=np.uint8, count=nbits)


def hf01_encode(source):
    freq = Counter(source)
    if len(source) > 0xffffffff:
        raise ValueError('Message too long')
    codes = _codes(_tree(freq))
    bits = [bit for byte in source for bit in codes[byte]]
    if len(bits) > 0xffffffff:
        raise ValueError('Huffman bits too long')
    header = bytearray(HF_HDR.pack(b'HF01', len(source), len(freq)))
    for s in sorted(freq):
        header.extend(struct.pack('<BI', s, freq[s]))
    header.extend(struct.pack('<I', len(bits)))
    return bytes(header) + pack_msb(bits)


def hf01_decode(stream):
    if len(stream) < HF_HDR.size + 4:
        raise ValueError('Truncated HF01')
    magic, original, nsym = HF_HDR.unpack_from(stream)
    if magic != b'HF01' or nsym > 256:
        raise ValueError('Bad HF01 header')
    pos = HF_HDR.size
    if len(stream) < pos + 5*nsym + 4:
        raise ValueError('Truncated HF01 frequency table')
    freq = {}
    for _ in range(nsym):
        sym, n = struct.unpack_from('<BI', stream, pos)
        pos += 5
        if sym in freq or n == 0:
            raise ValueError('Invalid symbol frequency')
        freq[sym] = n
    bit_count, = struct.unpack_from('<I', stream, pos)
    pos += 4
    if sum(freq.values()) != original or bit_count > 8*(len(stream)-pos):
        raise ValueError('Incorrect HF01 counts')
    if len(stream)-pos != math.ceil(bit_count/8):
        raise ValueError('Unexpected HF01 padding/trailing bytes')
    if original == 0:
        if nsym != 0 or bit_count != 0:
            raise ValueError('Invalid empty HF01')
        return b''
    root = _tree(freq)
    if isinstance(root, int):
        if bit_count != original:
            raise ValueError('Invalid one-symbol bit count')
        return bytes([root])*original
    out = bytearray()
    node = root
    for k in range(bit_count):
        bit = (stream[pos+k//8] >> (7-k%8)) & 1
        node = node[bit]
        if isinstance(node, int):
            out.append(node)
            if len(out) > original:
                raise ValueError('HF01 output overflow')
            node = root
    if len(out) != original or node is not root:
        raise ValueError('HF01 incomplete decode')
    return bytes(out)


def canonical_image(array):
    a = np.asarray(array)
    if a.dtype != np.uint8 or a.shape not in ((512,512),(512,512,3)):
        raise ValueError('Expected uint8 512x512 L or RGB image')
    return a


def groups_for_payload(nbits, channels):
    count = math.ceil(nbits/64)
    if count > 4096*channels:
        raise ValueError('One-LSB capacity exceeded')
    result = []
    for g in range(count):
        spatial = g // channels
        ch = g % channels + 1
        block_r, block_c = divmod(spatial, 64)
        rnd = (block_r // 8)*8 + (block_c//8) + 1
        brnd = (block_r % 8)*8 + (block_c%8) + 1
        result.append((rnd,brnd,ch))
    return result


def indices_for_groups(groups, channels, nbits):
    indices=[]
    seen=set()
    for rnd,brnd,ch in groups:
        if not (1<=rnd<=64 and 1<=brnd<=64 and 1<=ch<=channels):
            raise ValueError('Out-of-bounds SK04 group')
        top_r,top_c=divmod(rnd-1,8)
        sub_r,sub_c=divmod(brnd-1,8)
        row0=(top_r*8+sub_r)*8
        col0=(top_c*8+sub_c)*8
        for r in range(row0,row0+8):
            for c in range(col0,col0+8):
                idx = r + c*512 + (ch-1)*512*512
                if idx in seen:
                    raise ValueError('Duplicate carrier location')
                seen.add(idx)
                indices.append(idx)
    if len(indices)<nbits:
        raise ValueError('Insufficient SK04 locations')
    return np.array(indices[:nbits],dtype=np.int64)


def sk04_pack(shape, nbits, compressed, source, groups, flags):
    h,w=shape[:2]; ch=shape[2] if len(shape)==3 else 1
    if len(groups) != math.ceil(nbits/64) or len(flags)!=len(groups):
        raise ValueError('Key group count mismatch')
    hdr=SK_HDR.pack(b'SK04',1,h,w,ch,64,nbits,len(compressed),len(source),
        zlib.crc32(source),zlib.crc32(compressed),len(groups))
    return hdr + bytes(v for triple in groups for v in triple) + pack_lsb(flags)


def sk04_unpack(data):
    if len(data)<SK_HDR.size:
        raise ValueError('Short SK04')
    magic,version,h,w,ch,group_size,nbits,nbytes,source_bytes,source_crc,compressed_crc,n = SK_HDR.unpack_from(data)
    if magic != b'SK04' or version!=1 or h!=512 or w!=512 or ch not in (1,3) or group_size!=64:
        raise ValueError('Invalid SK04 header')
    if nbits > h*w*ch or nbytes*8 != nbits or n != math.ceil(nbits/64):
        raise ValueError('Inconsistent SK04 capacity/size')
    expected=SK_HDR.size+3*n+math.ceil(n/8)
    if len(data)!=expected:
        raise ValueError('Truncated/extra SK04 bytes')
    group_b=data[SK_HDR.size:SK_HDR.size+3*n]
    groups=[tuple(group_b[i:i+3]) for i in range(0,len(group_b),3)]
    flags=unpack_lsb(data[SK_HDR.size+3*n:],n)
    return dict(shape=(h,w) if ch==1 else (h,w,ch), channels=ch, nbits=nbits,
        compressed_bytes=nbytes, source_bytes=source_bytes,source_crc=source_crc,
        compressed_crc=compressed_crc,groups=groups,flags=flags)


def embed(cover,source):
    cover=canonical_image(cover)
    stream=hf01_encode(source)
    bits=unpack_lsb(stream)
    if len(bits) > cover.size:
        raise ValueError('Compressed stream over capacity')
    ch=1 if cover.ndim==2 else 3
    groups=groups_for_payload(len(bits),ch)
    idx=indices_for_groups(groups,ch,len(bits))
    carrier=cover.flatten(order='F').copy()
    flags=[]; direct=0;chosen=0
    for g in range(len(groups)):
        a=g*64;b=min(a+64,len(bits));sl=idx[a:b];raw=bits[a:b]
        old=carrier[sl] & 1
        c_direct=int(np.count_nonzero(old!=raw))
        c_inverse=(b-a)-c_direct
        invert=c_inverse<c_direct
        flags.append(invert)
        direct+=c_direct;chosen+=min(c_direct,c_inverse)
        target=1-raw if invert else raw
        carrier[sl] = (carrier[sl] & 0xfe) | target
    key=sk04_pack(cover.shape,len(bits),stream,source,groups,flags)
    stego=carrier.reshape(cover.shape,order='F')
    return stego,key,dict(source_bytes=len(source),embedded_bytes=len(stream),
        embedded_bits=len(bits), key_bytes=len(key), direct_changes=direct,
        actual_changes=chosen, image_samples=cover.size,
        key_bpp=len(key)*8/(512*512))


def extract(stego,key_bytes):
    stego=canonical_image(stego)
    key=sk04_unpack(key_bytes)
    if stego.shape!=key['shape']:
        raise ValueError('Stego/key dimensions differ')
    idx=indices_for_groups(key['groups'],key['channels'],key['nbits'])
    bits=stego.flatten(order='F')[idx]&1
    bits=bits.copy()
    for g,flag in enumerate(key['flags']):
        if flag:
            a=g*64;b=min(a+64,len(bits));bits[a:b]^=1
    stream=pack_lsb(bits)
    if len(stream)!=key['compressed_bytes'] or zlib.crc32(stream)!=key['compressed_crc']:
        raise ValueError('Compressed-stream CRC mismatch')
    source=hf01_decode(stream)
    if len(source)!=key['source_bytes'] or zlib.crc32(source)!=key['source_crc']:
        raise ValueError('Source CRC/length mismatch')
    return source


def image_load(path):
    with Image.open(path) as img:
        if img.mode not in ('L','RGB'):
            raise ValueError('Unsupported image mode')
        return canonical_image(np.asarray(img).copy())


def image_save(path,arr):
    Image.fromarray(canonical_image(arr)).save(path)
