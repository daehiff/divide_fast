OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
