OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[8];
rx(pi) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
cx q[7],q[1];
rx(pi) q[3];
cx q[4],q[0];
rz(7*pi/4) q[0];
cx q[4],q[0];
rz(5*pi/4) q[1];
rx(-3*pi/2) q[7];
rx(11*pi/4) q[5];
rx(5*pi/4) q[8];
rz(-pi/4) q[7];
rz(pi) q[8];
rx(pi) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
cx q[7],q[1];
rx(pi) q[3];
cx q[4],q[0];
rz(7*pi/4) q[0];
cx q[4],q[0];
rz(5*pi/4) q[1];
rx(-3*pi/2) q[7];
rx(11*pi/4) q[5];
rx(5*pi/4) q[8];
rz(-pi/4) q[7];
rz(pi) q[8];
rx(pi) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
cx q[7],q[1];
rx(pi) q[3];
cx q[4],q[0];
rz(7*pi/4) q[0];
cx q[4],q[0];
rz(5*pi/4) q[1];
rx(-3*pi/2) q[7];
rx(11*pi/4) q[5];
rx(5*pi/4) q[8];
rz(-pi/4) q[7];
rz(pi) q[8];
rx(pi) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
cx q[7],q[1];
rx(pi) q[3];
cx q[4],q[0];
rz(7*pi/4) q[0];
cx q[4],q[0];
rz(5*pi/4) q[1];
rx(-3*pi/2) q[7];
rx(11*pi/4) q[5];
rx(5*pi/4) q[8];
rz(-pi/4) q[7];
rz(pi) q[8];
rx(pi) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
cx q[7],q[1];
rx(pi) q[3];
cx q[4],q[0];
rz(7*pi/4) q[0];
cx q[4],q[0];
rz(5*pi/4) q[1];
rx(-3*pi/2) q[7];
rx(11*pi/4) q[5];
rx(5*pi/4) q[8];
rz(-pi/4) q[7];