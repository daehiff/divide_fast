OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rx(pi/2) q[3];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[0],q[3];
rx(pi/2) q[0];
cx q[0],q[3];
rz(pi/2) q[3];
rz(pi/2) q[1];
