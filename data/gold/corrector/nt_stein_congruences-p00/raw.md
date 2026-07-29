We use the is_prime function to make a table of the first few Mersenne primes (see Section 1.2.3).

sage: for p in primes(100):
...    if is_prime(2^p - 1):
...    print p, 2^p - 1
2 3
3 7
5 31
7 127
13 8191
17 131071
19 524287
31 2147483647
61 2305843009213693951
89 61897001964269013744956211

There is a specialized test for primality of Mersenne numbers called the Lucas-Lehmer test. This remarkably simple algorithm determines provably correctly whether or not a number  \( 2^{p}-1 \)  is prime. We implement it in a few lines of code and use the Lucas-Lehmer test to check for primality of two Mersenne numbers:

sage: def is_prime_lucas_lehmer(p):
...    s = Mod(4, 2^p - 1)
...    for i in range(3, p+1):
...    s = s^2 - 2
...    return s == 0
sage: # Check primality of 2^9941 - 1
sage: is_prime_lucas_lehmer(9941)
True
sage: # Check primality of 2^next_prime(1000)-1
sage: is_prime_lucas_lehmer(next_prime(1000))
False

For more on Mersenne primes, see the Great Internet Mersenne Prime Search (GIMPS) project at http://www.mersenne.org/.

### 2.5 The Structure of  \( (\mathbf{Z}/p\mathbf{Z})^{*} \)

This section is about the structure of the group \((\mathbf{Z} / p\mathbf{Z})^{*}\) of units modulo a prime number \(p\). The main result is that this group is always cyclic. We will use this result later in Chapter 4 in our proof of quadratic reciprocity.

Definition 2.5.1 (Primitive Root). A primitive root modulo an integer \( n \) is an element of \( (\mathbf{Z} / n\mathbf{Z})^* \) of order \( \varphi(n) \).