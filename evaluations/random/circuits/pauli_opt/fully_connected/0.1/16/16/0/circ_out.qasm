OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7],q[15];
cx q[15],q[7];
rz(pi) q[7];
cx q[15],q[7];
rz(pi) q[6];
rz(7*pi/4) q[5];
rz(pi/2) q[4];
rz(7*pi/4) q[7];
rz(7*pi/4) q[9];
rx(5*pi/4) q[8];
rx(3*pi/2) q[11];
rx(3*pi/2) q[12];
rx(3*pi/2) q[0];
rx(5*pi/4) q[1];
rx(3*pi/4) q[10];
rz(pi/2) q[12];
rz(5*pi/4) q[0];
cx q[15],q[7];
rz(pi) q[7];
cx q[15],q[7];
rz(pi) q[6];
rz(7*pi/4) q[5];
rz(pi/2) q[4];
rz(7*pi/4) q[7];
rz(7*pi/4) q[9];
rx(5*pi/4) q[8];
rx(3*pi/2) q[11];
rx(3*pi/2) q[12];
rx(3*pi/2) q[0];
rx(5*pi/4) q[1];
rx(3*pi/4) q[10];
rz(pi/2) q[12];
rz(5*pi/4) q[0];
cx q[15],q[7];
rz(pi) q[7];
cx q[15],q[7];
rz(pi) q[6];
rz(7*pi/4) q[5];
rz(pi/2) q[4];
rz(7*pi/4) q[7];
rz(7*pi/4) q[9];
rx(5*pi/4) q[8];
rx(3*pi/2) q[11];
rx(3*pi/2) q[12];
rx(3*pi/2) q[0];
rx(5*pi/4) q[1];
rx(3*pi/4) q[10];
rz(pi/2) q[12];
rz(5*pi/4) q[0];
cx q[15],q[7];
rz(pi) q[7];
cx q[15],q[7];
rz(pi) q[6];
rz(7*pi/4) q[5];
rz(pi/2) q[4];
rz(7*pi/4) q[7];
rz(7*pi/4) q[9];
rx(5*pi/4) q[8];
rx(3*pi/2) q[11];
rx(3*pi/2) q[12];
rx(3*pi/2) q[0];
rx(5*pi/4) q[1];
rx(3*pi/4) q[10];
rz(pi/2) q[12];
rz(5*pi/4) q[0];
cx q[15],q[7];
rz(pi) q[7];
cx q[15],q[7];
rz(pi) q[6];
rz(7*pi/4) q[5];
rz(pi/2) q[4];
rz(7*pi/4) q[7];
rz(7*pi/4) q[9];
rx(5*pi/4) q[8];
rx(3*pi/2) q[11];
rx(3*pi/2) q[12];
rx(3*pi/2) q[0];
rx(5*pi/4) q[1];
rx(3*pi/4) q[10];
rz(pi/2) q[12];
rz(5*pi/4) q[0];
cx q[7],q[15];
