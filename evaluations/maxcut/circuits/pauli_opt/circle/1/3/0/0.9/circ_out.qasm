OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(pi/2) q[2];
rx(pi/2) q[1];
rx(pi/2) q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[2],q[0];
rz(pi/2) q[0];
cx q[2],q[0];
