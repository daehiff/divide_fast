OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(pi) q[9];
rz(pi) q[14];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[6];
rx(pi) q[9];
rx(pi) q[12];
rx(pi) q[14];
rz(pi/2) q[5];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/4) q[15];
rz(3*pi/2) q[8];
rz(pi/2) q[7];
rx(7*pi/4) q[9];
rx(pi/4) q[4];
rx(5*pi/4) q[5];
rx(3*pi/2) q[14];
rx(7*pi/4) q[13];
rx(7*pi/4) q[12];
rx(pi/2) q[2];
rx(7*pi/4) q[3];
rx(pi/2) q[0];
rx(pi/4) q[7];
rx(5*pi/4) q[10];
rx(3*pi/4) q[1];
rx(3*pi/2) q[11];
rx(7*pi/4) q[15];
rx(7*pi/4) q[8];
rz(pi/4) q[3];
rz(7*pi/4) q[9];
rz(3*pi/2) q[11];
rz(3*pi/4) q[4];
rz(7*pi/4) q[5];
rz(5*pi/4) q[13];
rz(5*pi/4) q[14];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/4) q[0];
rz(7*pi/4) q[15];
rz(3*pi/4) q[7];
rz(7*pi/4) q[8];
rx(7*pi/4) q[4];
rx(3*pi/4) q[13];
rx(3*pi/2) q[9];
rx(7*pi/4) q[5];
rx(5*pi/4) q[14];
rx(pi/2) q[3];
rx(pi/4) q[8];
rx(7*pi/4) q[0];
rx(7*pi/4) q[15];
rz(7*pi/4) q[13];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(3*pi/4) q[3];
rz(pi/4) q[9];
rx(3*pi/2) q[14];
rx(3*pi/4) q[13];
rx(7*pi/4) q[4];
rz(pi/2) q[4];
rz(5*pi/4) q[13];
rx(3*pi/2) q[13];
rz(pi) q[9];
rz(pi) q[14];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[6];
rx(pi) q[9];
rx(pi) q[12];
rx(pi) q[14];
rz(pi/2) q[5];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/4) q[15];
rz(3*pi/2) q[8];
rz(pi/2) q[7];
rx(7*pi/4) q[9];
rx(pi/4) q[4];
rx(5*pi/4) q[5];
rx(3*pi/2) q[14];
rx(7*pi/4) q[13];
rx(7*pi/4) q[12];
rx(pi/2) q[2];
rx(7*pi/4) q[3];
rx(pi/2) q[0];
rx(pi/4) q[7];
rx(5*pi/4) q[10];
rx(3*pi/4) q[1];
rx(3*pi/2) q[11];
rx(7*pi/4) q[15];
rx(7*pi/4) q[8];
rz(pi/4) q[3];
rz(7*pi/4) q[9];
rz(3*pi/2) q[11];
rz(3*pi/4) q[4];
rz(7*pi/4) q[5];
rz(5*pi/4) q[13];
rz(5*pi/4) q[14];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/4) q[0];
rz(7*pi/4) q[15];
rz(3*pi/4) q[7];
rz(7*pi/4) q[8];
rx(7*pi/4) q[4];
rx(3*pi/4) q[13];
rx(3*pi/2) q[9];
rx(7*pi/4) q[5];
rx(5*pi/4) q[14];
rx(pi/2) q[3];
rx(pi/4) q[8];
rx(7*pi/4) q[0];
rx(7*pi/4) q[15];
rz(7*pi/4) q[13];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(3*pi/4) q[3];
rz(pi/4) q[9];
rx(3*pi/2) q[14];
rx(3*pi/4) q[13];
rx(7*pi/4) q[4];
rz(pi/2) q[4];
rz(5*pi/4) q[13];
rx(3*pi/2) q[13];
rz(pi) q[9];
rz(pi) q[14];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[6];
rx(pi) q[9];
rx(pi) q[12];
rx(pi) q[14];
rz(pi/2) q[5];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/4) q[15];
rz(3*pi/2) q[8];
rz(pi/2) q[7];
rx(7*pi/4) q[9];
rx(pi/4) q[4];
rx(5*pi/4) q[5];
rx(3*pi/2) q[14];
rx(7*pi/4) q[13];
rx(7*pi/4) q[12];
rx(pi/2) q[2];
rx(7*pi/4) q[3];
rx(pi/2) q[0];
rx(pi/4) q[7];
rx(5*pi/4) q[10];
rx(3*pi/4) q[1];
rx(3*pi/2) q[11];
rx(7*pi/4) q[15];
rx(7*pi/4) q[8];
rz(pi/4) q[3];
rz(7*pi/4) q[9];
rz(3*pi/2) q[11];
rz(3*pi/4) q[4];
rz(7*pi/4) q[5];
rz(5*pi/4) q[13];
rz(5*pi/4) q[14];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/4) q[0];
rz(7*pi/4) q[15];
rz(3*pi/4) q[7];
rz(7*pi/4) q[8];
rx(7*pi/4) q[4];
rx(3*pi/4) q[13];
rx(3*pi/2) q[9];
rx(7*pi/4) q[5];
rx(5*pi/4) q[14];
rx(pi/2) q[3];
rx(pi/4) q[8];
rx(7*pi/4) q[0];
rx(7*pi/4) q[15];
rz(7*pi/4) q[13];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(3*pi/4) q[3];
rz(pi/4) q[9];
rx(3*pi/2) q[14];
rx(3*pi/4) q[13];
rx(7*pi/4) q[4];
rz(pi/2) q[4];
rz(5*pi/4) q[13];
rx(3*pi/2) q[13];
rz(pi) q[9];
rz(pi) q[14];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[6];
rx(pi) q[9];
rx(pi) q[12];
rx(pi) q[14];
rz(pi/2) q[5];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/4) q[15];
rz(3*pi/2) q[8];
rz(pi/2) q[7];
rx(7*pi/4) q[9];
rx(pi/4) q[4];
rx(5*pi/4) q[5];
rx(3*pi/2) q[14];
rx(7*pi/4) q[13];
rx(7*pi/4) q[12];
rx(pi/2) q[2];
rx(7*pi/4) q[3];
rx(pi/2) q[0];
rx(pi/4) q[7];
rx(5*pi/4) q[10];
rx(3*pi/4) q[1];
rx(3*pi/2) q[11];
rx(7*pi/4) q[15];
rx(7*pi/4) q[8];
rz(pi/4) q[3];
rz(7*pi/4) q[9];
rz(3*pi/2) q[11];
rz(3*pi/4) q[4];
rz(7*pi/4) q[5];
rz(5*pi/4) q[13];
rz(5*pi/4) q[14];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/4) q[0];
rz(7*pi/4) q[15];
rz(3*pi/4) q[7];
rz(7*pi/4) q[8];
rx(7*pi/4) q[4];
rx(3*pi/4) q[13];
rx(3*pi/2) q[9];
rx(7*pi/4) q[5];
rx(5*pi/4) q[14];
rx(pi/2) q[3];
rx(pi/4) q[8];
rx(7*pi/4) q[0];
rx(7*pi/4) q[15];
rz(7*pi/4) q[13];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(3*pi/4) q[3];
rz(pi/4) q[9];
rx(3*pi/2) q[14];
rx(3*pi/4) q[13];
rx(7*pi/4) q[4];
rz(pi/2) q[4];
rz(5*pi/4) q[13];
rx(3*pi/2) q[13];
rz(pi) q[9];
rz(pi) q[14];
rz(pi) q[15];
rx(pi) q[0];
rx(pi) q[6];
rx(pi) q[9];
rx(pi) q[12];
rx(pi) q[14];
rz(pi/2) q[5];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/4) q[15];
rz(3*pi/2) q[8];
rz(pi/2) q[7];
rx(7*pi/4) q[9];
rx(pi/4) q[4];
rx(5*pi/4) q[5];
rx(3*pi/2) q[14];
rx(7*pi/4) q[13];
rx(7*pi/4) q[12];
rx(pi/2) q[2];
rx(7*pi/4) q[3];
rx(pi/2) q[0];
rx(pi/4) q[7];
rx(5*pi/4) q[10];
rx(3*pi/4) q[1];
rx(3*pi/2) q[11];
rx(7*pi/4) q[15];
rx(7*pi/4) q[8];
rz(pi/4) q[3];
rz(7*pi/4) q[9];
rz(3*pi/2) q[11];
rz(3*pi/4) q[4];
rz(7*pi/4) q[5];
rz(5*pi/4) q[13];
rz(5*pi/4) q[14];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/4) q[0];
rz(7*pi/4) q[15];
rz(3*pi/4) q[7];
rz(7*pi/4) q[8];
rx(7*pi/4) q[4];
rx(3*pi/4) q[13];
rx(3*pi/2) q[9];
rx(7*pi/4) q[5];
rx(5*pi/4) q[14];
rx(pi/2) q[3];
rx(pi/4) q[8];
rx(7*pi/4) q[0];
rx(7*pi/4) q[15];
rz(7*pi/4) q[13];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(3*pi/4) q[3];
rz(pi/4) q[9];
rx(3*pi/2) q[14];
rx(3*pi/4) q[13];
rx(7*pi/4) q[4];
rz(pi/2) q[4];
rz(5*pi/4) q[13];
rx(3*pi/2) q[13];
