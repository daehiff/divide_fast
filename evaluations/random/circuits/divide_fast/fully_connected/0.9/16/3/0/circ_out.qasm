OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(pi) q[0];
rx(pi) q[0];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
rx(-pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
rz(11*pi/4) q[1];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[2];
rz(pi) q[0];
rx(pi) q[0];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
rx(-pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
rz(11*pi/4) q[1];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[2];
rz(pi) q[0];
rx(pi) q[0];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
rx(-pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
rz(11*pi/4) q[1];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[2];
rz(pi) q[0];
rx(pi) q[0];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
rx(-pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
rz(11*pi/4) q[1];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[2];
rz(pi) q[0];
rx(pi) q[0];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
rx(-pi/4) q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
rz(11*pi/4) q[1];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[2];
