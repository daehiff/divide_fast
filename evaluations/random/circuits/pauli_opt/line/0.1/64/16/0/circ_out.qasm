OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
rz(pi) q[6];
rz(pi) q[10];
rz(pi) q[12];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[7];
rx(pi) q[11];
rz(pi/4) q[3];
rz(5*pi/4) q[0];
rz(5*pi/4) q[5];
rz(pi/2) q[6];
rz(pi/4) q[4];
rx(3*pi/4) q[10];
rx(pi/2) q[9];
rx(pi/4) q[7];
rx(7*pi/4) q[15];
rx(3*pi/4) q[12];
rx(7*pi/4) q[13];
rx(3*pi/4) q[6];
rx(7*pi/4) q[8];
rx(3*pi/4) q[0];
rx(3*pi/4) q[14];
rx(5*pi/4) q[4];
rz(pi/2) q[10];
rz(pi/4) q[7];
rz(3*pi/4) q[15];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
rz(pi/4) q[9];
rz(3*pi/2) q[8];
rx(7*pi/4) q[10];
rx(3*pi/2) q[0];
rx(pi/2) q[8];
rz(3*pi/2) q[10];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
rz(pi) q[6];
rz(pi) q[10];
rz(pi) q[12];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[7];
rx(pi) q[11];
rz(pi/4) q[3];
rz(5*pi/4) q[0];
rz(5*pi/4) q[5];
rz(pi/2) q[6];
rz(pi/4) q[4];
rx(3*pi/4) q[10];
rx(pi/2) q[9];
rx(pi/4) q[7];
rx(7*pi/4) q[15];
rx(3*pi/4) q[12];
rx(7*pi/4) q[13];
rx(3*pi/4) q[6];
rx(7*pi/4) q[8];
rx(3*pi/4) q[0];
rx(3*pi/4) q[14];
rx(5*pi/4) q[4];
rz(pi/2) q[10];
rz(pi/4) q[7];
rz(3*pi/4) q[15];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
rz(pi/4) q[9];
rz(3*pi/2) q[8];
rx(7*pi/4) q[10];
rx(3*pi/2) q[0];
rx(pi/2) q[8];
rz(3*pi/2) q[10];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
rz(pi) q[6];
rz(pi) q[10];
rz(pi) q[12];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[7];
rx(pi) q[11];
rz(pi/4) q[3];
rz(5*pi/4) q[0];
rz(5*pi/4) q[5];
rz(pi/2) q[6];
rz(pi/4) q[4];
rx(3*pi/4) q[10];
rx(pi/2) q[9];
rx(pi/4) q[7];
rx(7*pi/4) q[15];
rx(3*pi/4) q[12];
rx(7*pi/4) q[13];
rx(3*pi/4) q[6];
rx(7*pi/4) q[8];
rx(3*pi/4) q[0];
rx(3*pi/4) q[14];
rx(5*pi/4) q[4];
rz(pi/2) q[10];
rz(pi/4) q[7];
rz(3*pi/4) q[15];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
rz(pi/4) q[9];
rz(3*pi/2) q[8];
rx(7*pi/4) q[10];
rx(3*pi/2) q[0];
rx(pi/2) q[8];
rz(3*pi/2) q[10];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
rz(pi) q[6];
rz(pi) q[10];
rz(pi) q[12];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[7];
rx(pi) q[11];
rz(pi/4) q[3];
rz(5*pi/4) q[0];
rz(5*pi/4) q[5];
rz(pi/2) q[6];
rz(pi/4) q[4];
rx(3*pi/4) q[10];
rx(pi/2) q[9];
rx(pi/4) q[7];
rx(7*pi/4) q[15];
rx(3*pi/4) q[12];
rx(7*pi/4) q[13];
rx(3*pi/4) q[6];
rx(7*pi/4) q[8];
rx(3*pi/4) q[0];
rx(3*pi/4) q[14];
rx(5*pi/4) q[4];
rz(pi/2) q[10];
rz(pi/4) q[7];
rz(3*pi/4) q[15];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
rz(pi/4) q[9];
rz(3*pi/2) q[8];
rx(7*pi/4) q[10];
rx(3*pi/2) q[0];
rx(pi/2) q[8];
rz(3*pi/2) q[10];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
rz(pi) q[6];
rz(pi) q[10];
rz(pi) q[12];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[7];
rx(pi) q[11];
rz(pi/4) q[3];
rz(5*pi/4) q[0];
rz(5*pi/4) q[5];
rz(pi/2) q[6];
rz(pi/4) q[4];
rx(3*pi/4) q[10];
rx(pi/2) q[9];
rx(pi/4) q[7];
rx(7*pi/4) q[15];
rx(3*pi/4) q[12];
rx(7*pi/4) q[13];
rx(3*pi/4) q[6];
rx(7*pi/4) q[8];
rx(3*pi/4) q[0];
rx(3*pi/4) q[14];
rx(5*pi/4) q[4];
rz(pi/2) q[10];
rz(pi/4) q[7];
rz(3*pi/4) q[15];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
rz(pi/4) q[9];
rz(3*pi/2) q[8];
rx(7*pi/4) q[10];
rx(3*pi/2) q[0];
rx(pi/2) q[8];
rz(3*pi/2) q[10];
