OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(pi) q[7];
rz(pi) q[9];
rz(pi) q[11];
rz(pi) q[15];
rx(pi) q[12];
rx(pi) q[13];
rz(3*pi/2) q[3];
rz(3*pi/4) q[8];
rz(3*pi/4) q[0];
rz(3*pi/2) q[2];
rz(7*pi/4) q[13];
rz(pi/4) q[12];
rz(3*pi/2) q[5];
rz(pi/4) q[1];
rz(pi/4) q[14];
rz(7*pi/4) q[6];
rx(7*pi/4) q[4];
rx(pi/4) q[0];
rx(pi/2) q[7];
rx(pi/2) q[3];
rx(3*pi/4) q[11];
rx(3*pi/2) q[10];
rx(3*pi/2) q[8];
rx(5*pi/4) q[5];
rx(pi/2) q[2];
rz(3*pi/4) q[8];
rz(3*pi/4) q[4];
rz(3*pi/2) q[10];
rz(3*pi/4) q[3];
rx(3*pi/4) q[3];
rx(3*pi/2) q[10];
rz(3*pi/2) q[3];
rx(3*pi/2) q[3];
rz(pi) q[7];
rz(pi) q[9];
rz(pi) q[11];
rz(pi) q[15];
rx(pi) q[12];
rx(pi) q[13];
rz(3*pi/2) q[3];
rz(3*pi/4) q[8];
rz(3*pi/4) q[0];
rz(3*pi/2) q[2];
rz(7*pi/4) q[13];
rz(pi/4) q[12];
rz(3*pi/2) q[5];
rz(pi/4) q[1];
rz(pi/4) q[14];
rz(7*pi/4) q[6];
rx(7*pi/4) q[4];
rx(pi/4) q[0];
rx(pi/2) q[7];
rx(pi/2) q[3];
rx(3*pi/4) q[11];
rx(3*pi/2) q[10];
rx(3*pi/2) q[8];
rx(5*pi/4) q[5];
rx(pi/2) q[2];
rz(3*pi/4) q[8];
rz(3*pi/4) q[4];
rz(3*pi/2) q[10];
rz(3*pi/4) q[3];
rx(3*pi/4) q[3];
rx(3*pi/2) q[10];
rz(3*pi/2) q[3];
rx(3*pi/2) q[3];
rz(pi) q[7];
rz(pi) q[9];
rz(pi) q[11];
rz(pi) q[15];
rx(pi) q[12];
rx(pi) q[13];
rz(3*pi/2) q[3];
rz(3*pi/4) q[8];
rz(3*pi/4) q[0];
rz(3*pi/2) q[2];
rz(7*pi/4) q[13];
rz(pi/4) q[12];
rz(3*pi/2) q[5];
rz(pi/4) q[1];
rz(pi/4) q[14];
rz(7*pi/4) q[6];
rx(7*pi/4) q[4];
rx(pi/4) q[0];
rx(pi/2) q[7];
rx(pi/2) q[3];
rx(3*pi/4) q[11];
rx(3*pi/2) q[10];
rx(3*pi/2) q[8];
rx(5*pi/4) q[5];
rx(pi/2) q[2];
rz(3*pi/4) q[8];
rz(3*pi/4) q[4];
rz(3*pi/2) q[10];
rz(3*pi/4) q[3];
rx(3*pi/4) q[3];
rx(3*pi/2) q[10];
rz(3*pi/2) q[3];
rx(3*pi/2) q[3];
rz(pi) q[7];
rz(pi) q[9];
rz(pi) q[11];
rz(pi) q[15];
rx(pi) q[12];
rx(pi) q[13];
rz(3*pi/2) q[3];
rz(3*pi/4) q[8];
rz(3*pi/4) q[0];
rz(3*pi/2) q[2];
rz(7*pi/4) q[13];
rz(pi/4) q[12];
rz(3*pi/2) q[5];
rz(pi/4) q[1];
rz(pi/4) q[14];
rz(7*pi/4) q[6];
rx(7*pi/4) q[4];
rx(pi/4) q[0];
rx(pi/2) q[7];
rx(pi/2) q[3];
rx(3*pi/4) q[11];
rx(3*pi/2) q[10];
rx(3*pi/2) q[8];
rx(5*pi/4) q[5];
rx(pi/2) q[2];
rz(3*pi/4) q[8];
rz(3*pi/4) q[4];
rz(3*pi/2) q[10];
rz(3*pi/4) q[3];
rx(3*pi/4) q[3];
rx(3*pi/2) q[10];
rz(3*pi/2) q[3];
rx(3*pi/2) q[3];
rz(pi) q[7];
rz(pi) q[9];
rz(pi) q[11];
rz(pi) q[15];
rx(pi) q[12];
rx(pi) q[13];
rz(3*pi/2) q[3];
rz(3*pi/4) q[8];
rz(3*pi/4) q[0];
rz(3*pi/2) q[2];
rz(7*pi/4) q[13];
rz(pi/4) q[12];
rz(3*pi/2) q[5];
rz(pi/4) q[1];
rz(pi/4) q[14];
rz(7*pi/4) q[6];
rx(7*pi/4) q[4];
rx(pi/4) q[0];
rx(pi/2) q[7];
rx(pi/2) q[3];
rx(3*pi/4) q[11];
rx(3*pi/2) q[10];
rx(3*pi/2) q[8];
rx(5*pi/4) q[5];
rx(pi/2) q[2];
rz(3*pi/4) q[8];
rz(3*pi/4) q[4];
rz(3*pi/2) q[10];
rz(3*pi/4) q[3];
rx(3*pi/4) q[3];
rx(3*pi/2) q[10];
rz(3*pi/2) q[3];
rx(3*pi/2) q[3];
