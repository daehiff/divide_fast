OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0],q[2];
cx q[5],q[4];
cx q[6],q[2];
rz(pi) q[2];
cx q[6],q[2];
cx q[6],q[0];
rz(5*pi/4) q[0];
cx q[6],q[0];
rz(5*pi/4) q[0];
rz(pi/4) q[2];
rx(pi/2) q[4];
rx(7*pi/4) q[8];
cx q[6],q[4];
rz(3*pi/2) q[4];
cx q[6],q[4];
rz(5*pi/4) q[4];
cx q[6],q[2];
rz(pi) q[2];
cx q[6],q[2];
cx q[6],q[0];
rz(5*pi/4) q[0];
cx q[6],q[0];
rz(5*pi/4) q[0];
rz(pi/4) q[2];
rx(pi/2) q[4];
rx(7*pi/4) q[8];
cx q[6],q[4];
rz(3*pi/2) q[4];
cx q[6],q[4];
rz(5*pi/4) q[4];
cx q[6],q[2];
rz(pi) q[2];
cx q[6],q[2];
cx q[6],q[0];
rz(5*pi/4) q[0];
cx q[6],q[0];
rz(5*pi/4) q[0];
rz(pi/4) q[2];
rx(pi/2) q[4];
rx(7*pi/4) q[8];
cx q[6],q[4];
rz(3*pi/2) q[4];
cx q[6],q[4];
rz(5*pi/4) q[4];
cx q[6],q[2];
rz(pi) q[2];
cx q[6],q[2];
cx q[6],q[0];
rz(5*pi/4) q[0];
cx q[6],q[0];
rz(5*pi/4) q[0];
rz(pi/4) q[2];
rx(pi/2) q[4];
rx(7*pi/4) q[8];
cx q[6],q[4];
rz(3*pi/2) q[4];
cx q[6],q[4];
rz(5*pi/4) q[4];
cx q[6],q[2];
rz(pi) q[2];
cx q[6],q[2];
cx q[6],q[0];
rz(5*pi/4) q[0];
cx q[6],q[0];
rz(5*pi/4) q[0];
rz(pi/4) q[2];
rx(pi/2) q[4];
rx(7*pi/4) q[8];
cx q[6],q[4];
rz(3*pi/2) q[4];
cx q[6],q[4];
rz(5*pi/4) q[4];
cx q[0],q[2];
cx q[5],q[4];
