OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7],q[10];
cx q[3],q[10];
cx q[3],q[7];
cx q[3],q[6];
cx q[1],q[14];
rz(pi) q[1];
rz(pi) q[3];
rz(pi) q[9];
rz(pi) q[13];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[8];
rx(pi) q[9];
rx(pi) q[14];
cx q[14],q[3];
cx q[9],q[15];
cx q[5],q[13];
cx q[1],q[8];
cx q[5],q[14];
rx(pi/4) q[5];
cx q[5],q[14];
rx(pi) q[15];
rz(3*pi/4) q[0];
cx q[1],q[4];
cx q[1],q[3];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[1],q[10];
cx q[1],q[2];
cx q[1],q[12];
rx(-5*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[2];
cx q[1],q[10];
cx q[1],q[5];
cx q[7],q[9];
cx q[1],q[8];
cx q[7],q[8];
cx q[7],q[13];
cx q[7],q[15];
rx(-5*pi/4) q[7];
cx q[7],q[15];
cx q[7],q[13];
cx q[7],q[8];
rz(-3*pi/4) q[4];
cx q[0],q[7];
rx(3*pi/2) q[0];
cx q[0],q[7];
rz(3*pi/4) q[8];
cx q[15],q[1];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[15],q[1];
cx q[1],q[9];
rx(3*pi/2) q[1];
cx q[1],q[9];
cx q[14],q[3];
cx q[9],q[15];
cx q[7],q[15];
cx q[7],q[9];
cx q[5],q[13];
rz(pi) q[1];
rz(pi) q[3];
rz(pi) q[9];
rz(pi) q[13];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[8];
rx(pi) q[9];
rx(pi) q[14];
cx q[14],q[3];
cx q[9],q[15];
cx q[5],q[13];
cx q[1],q[8];
cx q[5],q[14];
rx(pi/4) q[5];
cx q[5],q[14];
rx(pi) q[15];
rz(3*pi/4) q[0];
cx q[1],q[4];
cx q[1],q[3];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[1],q[10];
cx q[1],q[2];
cx q[1],q[12];
rx(-5*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[2];
cx q[1],q[10];
cx q[1],q[5];
cx q[7],q[9];
cx q[1],q[8];
cx q[7],q[8];
cx q[7],q[13];
cx q[7],q[15];
rx(-5*pi/4) q[7];
cx q[7],q[15];
cx q[7],q[13];
cx q[7],q[8];
rz(-3*pi/4) q[4];
cx q[0],q[7];
rx(3*pi/2) q[0];
cx q[0],q[7];
rz(3*pi/4) q[8];
cx q[15],q[1];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[15],q[1];
cx q[1],q[9];
rx(3*pi/2) q[1];
cx q[1],q[9];
cx q[14],q[3];
cx q[9],q[15];
cx q[7],q[15];
cx q[7],q[9];
cx q[5],q[13];
rz(pi) q[1];
rz(pi) q[3];
rz(pi) q[9];
rz(pi) q[13];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[8];
rx(pi) q[9];
rx(pi) q[14];
cx q[14],q[3];
cx q[9],q[15];
cx q[5],q[13];
cx q[1],q[8];
cx q[5],q[14];
rx(pi/4) q[5];
cx q[5],q[14];
rx(pi) q[15];
rz(3*pi/4) q[0];
cx q[1],q[4];
cx q[1],q[3];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[1],q[10];
cx q[1],q[2];
cx q[1],q[12];
rx(-5*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[2];
cx q[1],q[10];
cx q[1],q[5];
cx q[7],q[9];
cx q[1],q[8];
cx q[7],q[8];
cx q[7],q[13];
cx q[7],q[15];
rx(-5*pi/4) q[7];
cx q[7],q[15];
cx q[7],q[13];
cx q[7],q[8];
rz(-3*pi/4) q[4];
cx q[0],q[7];
rx(3*pi/2) q[0];
cx q[0],q[7];
rz(3*pi/4) q[8];
cx q[15],q[1];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[15],q[1];
cx q[1],q[9];
rx(3*pi/2) q[1];
cx q[1],q[9];
cx q[14],q[3];
cx q[9],q[15];
cx q[7],q[15];
cx q[7],q[9];
cx q[5],q[13];
rz(pi) q[1];
rz(pi) q[3];
rz(pi) q[9];
rz(pi) q[13];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[8];
rx(pi) q[9];
rx(pi) q[14];
cx q[14],q[3];
cx q[9],q[15];
cx q[5],q[13];
cx q[1],q[8];
cx q[5],q[14];
rx(pi/4) q[5];
cx q[5],q[14];
rx(pi) q[15];
rz(3*pi/4) q[0];
cx q[1],q[4];
cx q[1],q[3];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[1],q[10];
cx q[1],q[2];
cx q[1],q[12];
rx(-5*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[2];
cx q[1],q[10];
cx q[1],q[5];
cx q[7],q[9];
cx q[1],q[8];
cx q[7],q[8];
cx q[7],q[13];
cx q[7],q[15];
rx(-5*pi/4) q[7];
cx q[7],q[15];
cx q[7],q[13];
cx q[7],q[8];
rz(-3*pi/4) q[4];
cx q[0],q[7];
rx(3*pi/2) q[0];
cx q[0],q[7];
rz(3*pi/4) q[8];
cx q[15],q[1];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[15],q[1];
cx q[1],q[9];
rx(3*pi/2) q[1];
cx q[1],q[9];
cx q[14],q[3];
cx q[9],q[15];
cx q[7],q[15];
cx q[7],q[9];
cx q[5],q[13];
rz(pi) q[1];
rz(pi) q[3];
rz(pi) q[9];
rz(pi) q[13];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[8];
rx(pi) q[9];
rx(pi) q[14];
cx q[14],q[3];
cx q[9],q[15];
cx q[5],q[13];
cx q[1],q[8];
cx q[5],q[14];
rx(pi/4) q[5];
cx q[5],q[14];
rx(pi) q[15];
rz(3*pi/4) q[0];
cx q[1],q[4];
cx q[1],q[3];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[1],q[10];
cx q[1],q[2];
cx q[1],q[12];
rx(-5*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[2];
cx q[1],q[10];
cx q[1],q[5];
cx q[7],q[9];
cx q[1],q[8];
cx q[7],q[8];
cx q[7],q[13];
cx q[7],q[15];
rx(-5*pi/4) q[7];
cx q[7],q[15];
cx q[7],q[13];
cx q[7],q[8];
rz(-3*pi/4) q[4];
cx q[0],q[7];
rx(3*pi/2) q[0];
cx q[0],q[7];
rz(3*pi/4) q[8];
cx q[15],q[1];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[15],q[1];
cx q[1],q[9];
rx(3*pi/2) q[1];
cx q[1],q[9];
cx q[14],q[3];
cx q[9],q[15];
cx q[7],q[15];
cx q[7],q[9];
cx q[5],q[13];
cx q[7],q[10];
cx q[3],q[7];
cx q[3],q[6];
cx q[1],q[14];