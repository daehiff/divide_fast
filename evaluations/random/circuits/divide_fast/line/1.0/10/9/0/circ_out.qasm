OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[4];
rx(pi) q[2];
rx(pi) q[3];
rx(pi) q[5];
rx(pi) q[6];
rz(pi/4) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[5],q[3];
rz(pi/4) q[5];
cx q[5],q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
cx q[5],q[7];
rz(pi) q[4];
rx(pi) q[2];
rx(pi) q[3];
rx(pi) q[5];
rx(pi) q[6];
rz(pi/4) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[5],q[3];
rz(pi/4) q[5];
cx q[5],q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
cx q[5],q[7];
rz(pi) q[4];
rx(pi) q[2];
rx(pi) q[3];
rx(pi) q[5];
rx(pi) q[6];
rz(pi/4) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[5],q[3];
rz(pi/4) q[5];
cx q[5],q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
cx q[5],q[7];
rz(pi) q[4];
rx(pi) q[2];
rx(pi) q[3];
rx(pi) q[5];
rx(pi) q[6];
rz(pi/4) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[5],q[3];
rz(pi/4) q[5];
cx q[5],q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
cx q[5],q[7];
rz(pi) q[4];
rx(pi) q[2];
rx(pi) q[3];
rx(pi) q[5];
rx(pi) q[6];
rz(pi/4) q[0];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[5],q[3];
rz(pi/4) q[5];
cx q[5],q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
cx q[5],q[7];
