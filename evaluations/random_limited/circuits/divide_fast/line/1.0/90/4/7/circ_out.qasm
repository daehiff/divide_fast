OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cx q[3],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[2],q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
rx(pi) q[3];
cx q[2],q[1];
rz(-5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(-7*pi/4) q[2];
rz(pi) q[0];
cx q[2],q[1];
rz(-3*pi/4) q[1];
cx q[2],q[1];
rx(-pi/4) q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[1];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
rx(-7*pi/4) q[2];
rz(3*pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[0];
rx(-7*pi/4) q[0];
cx q[1],q[0];
rx(7*pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(-5*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[1];
rz(5*pi/4) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[0],q[1];
rx(-3*pi/2) q[0];
rz(3*pi/4) q[0];
rz(3*pi/4) q[3];
rx(-pi/2) q[2];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[3];
rx(5*pi/4) q[1];
rz(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
rx(-pi/4) q[2];
rx(pi/2) q[1];
rx(pi/4) q[3];
rx(-5*pi/4) q[0];
cx q[3],q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
rx(7*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(3*pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
cx q[2],q[3];
rx(pi/2) q[3];
rx(3*pi/2) q[2];
rx(3*pi/2) q[1];
rx(5*pi/4) q[0];
rz(7*pi/4) q[0];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
rx(pi/4) q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
rx(pi) q[3];
cx q[2],q[1];
rz(-5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(-7*pi/4) q[2];
rz(pi) q[0];
cx q[2],q[1];
rz(-3*pi/4) q[1];
cx q[2],q[1];
rx(-pi/4) q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[1];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
rx(-7*pi/4) q[2];
rz(3*pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[0];
rx(-7*pi/4) q[0];
cx q[1],q[0];
rx(7*pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(-5*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[1];
rz(5*pi/4) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[0],q[1];
rx(-3*pi/2) q[0];
rz(3*pi/4) q[0];
rz(3*pi/4) q[3];
rx(-pi/2) q[2];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[3];
rx(5*pi/4) q[1];
rz(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
rx(-pi/4) q[2];
rx(pi/2) q[1];
rx(pi/4) q[3];
rx(-5*pi/4) q[0];
cx q[3],q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
rx(7*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(3*pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
cx q[2],q[3];
rx(pi/2) q[3];
rx(3*pi/2) q[2];
rx(3*pi/2) q[1];
rx(5*pi/4) q[0];
rz(7*pi/4) q[0];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
rx(pi/4) q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
rx(pi) q[3];
cx q[2],q[1];
rz(-5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(-7*pi/4) q[2];
rz(pi) q[0];
cx q[2],q[1];
rz(-3*pi/4) q[1];
cx q[2],q[1];
rx(-pi/4) q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[1];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
rx(-7*pi/4) q[2];
rz(3*pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[0];
rx(-7*pi/4) q[0];
cx q[1],q[0];
rx(7*pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(-5*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[1];
rz(5*pi/4) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[0],q[1];
rx(-3*pi/2) q[0];
rz(3*pi/4) q[0];
rz(3*pi/4) q[3];
rx(-pi/2) q[2];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[3];
rx(5*pi/4) q[1];
rz(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
rx(-pi/4) q[2];
rx(pi/2) q[1];
rx(pi/4) q[3];
rx(-5*pi/4) q[0];
cx q[3],q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
rx(7*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(3*pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
cx q[2],q[3];
rx(pi/2) q[3];
rx(3*pi/2) q[2];
rx(3*pi/2) q[1];
rx(5*pi/4) q[0];
rz(7*pi/4) q[0];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
rx(pi/4) q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
rx(pi) q[3];
cx q[2],q[1];
rz(-5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(-7*pi/4) q[2];
rz(pi) q[0];
cx q[2],q[1];
rz(-3*pi/4) q[1];
cx q[2],q[1];
rx(-pi/4) q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[1];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
rx(-7*pi/4) q[2];
rz(3*pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[0];
rx(-7*pi/4) q[0];
cx q[1],q[0];
rx(7*pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(-5*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[1];
rz(5*pi/4) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[0],q[1];
rx(-3*pi/2) q[0];
rz(3*pi/4) q[0];
rz(3*pi/4) q[3];
rx(-pi/2) q[2];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[3];
rx(5*pi/4) q[1];
rz(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
rx(-pi/4) q[2];
rx(pi/2) q[1];
rx(pi/4) q[3];
rx(-5*pi/4) q[0];
cx q[3],q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
rx(7*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(3*pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
cx q[2],q[3];
rx(pi/2) q[3];
rx(3*pi/2) q[2];
rx(3*pi/2) q[1];
rx(5*pi/4) q[0];
rz(7*pi/4) q[0];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
rx(pi/4) q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
rx(pi) q[3];
cx q[2],q[1];
rz(-5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(-7*pi/4) q[2];
rz(pi) q[0];
cx q[2],q[1];
rz(-3*pi/4) q[1];
cx q[2],q[1];
rx(-pi/4) q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[1];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
rx(-7*pi/4) q[2];
rz(3*pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[0];
rx(-7*pi/4) q[0];
cx q[1],q[0];
rx(7*pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(-5*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[1];
rz(5*pi/4) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[0],q[1];
rx(-3*pi/2) q[0];
rz(3*pi/4) q[0];
rz(3*pi/4) q[3];
rx(-pi/2) q[2];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[3];
rx(5*pi/4) q[1];
rz(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
rx(-pi/4) q[2];
rx(pi/2) q[1];
rx(pi/4) q[3];
rx(-5*pi/4) q[0];
cx q[3],q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
rx(7*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(3*pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rx(3*pi/4) q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
cx q[2],q[3];
rx(pi/2) q[3];
rx(3*pi/2) q[2];
rx(3*pi/2) q[1];
rx(5*pi/4) q[0];
rz(7*pi/4) q[0];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
rx(pi/4) q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[2],q[1];
cx q[3],q[2];
