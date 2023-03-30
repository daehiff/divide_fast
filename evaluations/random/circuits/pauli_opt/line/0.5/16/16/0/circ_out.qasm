OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[10],q[9];
cx q[12],q[11];
cx q[6],q[5];
cx q[10],q[11];
cx q[2],q[1];
cx q[15],q[14];
cx q[8],q[7];
cx q[5],q[4];
cx q[1],q[0];
cx q[7],q[6];
cx q[11],q[10];
cx q[4],q[3];
rx(pi) q[2];
cx q[4],q[5];
cx q[3],q[4];
rx(pi) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[4],q[5];
rx(pi) q[4];
cx q[4],q[5];
cx q[7],q[8];
rx(pi) q[7];
cx q[7],q[8];
rx(pi) q[9];
rx(pi) q[11];
cx q[10],q[11];
cx q[9],q[10];
rx(pi) q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
rx(pi) q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[6],q[8];
cx q[5],q[6];
cx q[3],q[5];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[6],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[10],q[12];
cx q[9],q[10];
cx q[5],q[9];
rx(7*pi/4) q[5];
cx q[5],q[9];
cx q[9],q[10];
cx q[10],q[12];
cx q[14],q[15];
cx q[11],q[14];
cx q[5],q[11];
cx q[4],q[5];
cx q[3],q[4];
rx(7*pi/4) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/2) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[14],q[8];
cx q[15],q[14];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[10],q[6];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[6];
cx q[9],q[7];
cx q[10],q[9];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
rz(7*pi/4) q[13];
rz(5*pi/4) q[3];
cx q[13],q[9];
cx q[9],q[8];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(3*pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[6],q[4];
cx q[8],q[6];
cx q[9],q[8];
cx q[13],q[9];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[4],q[6];
cx q[2],q[4];
rx(7*pi/4) q[2];
cx q[2],q[4];
cx q[4],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[8],q[13];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[3];
rx(7*pi/4) q[1];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
rx(pi/4) q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[11],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[8],q[6];
cx q[11],q[8];
rx(pi) q[2];
cx q[4],q[5];
cx q[3],q[4];
rx(pi) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[4],q[5];
rx(pi) q[4];
cx q[4],q[5];
cx q[7],q[8];
rx(pi) q[7];
cx q[7],q[8];
rx(pi) q[9];
rx(pi) q[11];
cx q[10],q[11];
cx q[9],q[10];
rx(pi) q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
rx(pi) q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[6],q[8];
cx q[5],q[6];
cx q[3],q[5];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[6],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[10],q[12];
cx q[9],q[10];
cx q[5],q[9];
rx(7*pi/4) q[5];
cx q[5],q[9];
cx q[9],q[10];
cx q[10],q[12];
cx q[14],q[15];
cx q[11],q[14];
cx q[5],q[11];
cx q[4],q[5];
cx q[3],q[4];
rx(7*pi/4) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/2) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[14],q[8];
cx q[15],q[14];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[10],q[6];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[6];
cx q[9],q[7];
cx q[10],q[9];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
rz(7*pi/4) q[13];
rz(5*pi/4) q[3];
cx q[13],q[9];
cx q[9],q[8];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(3*pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[6],q[4];
cx q[8],q[6];
cx q[9],q[8];
cx q[13],q[9];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[4],q[6];
cx q[2],q[4];
rx(7*pi/4) q[2];
cx q[2],q[4];
cx q[4],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[8],q[13];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[3];
rx(7*pi/4) q[1];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
rx(pi/4) q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[11],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[8],q[6];
cx q[11],q[8];
rx(pi) q[2];
cx q[4],q[5];
cx q[3],q[4];
rx(pi) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[4],q[5];
rx(pi) q[4];
cx q[4],q[5];
cx q[7],q[8];
rx(pi) q[7];
cx q[7],q[8];
rx(pi) q[9];
rx(pi) q[11];
cx q[10],q[11];
cx q[9],q[10];
rx(pi) q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
rx(pi) q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[6],q[8];
cx q[5],q[6];
cx q[3],q[5];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[6],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[10],q[12];
cx q[9],q[10];
cx q[5],q[9];
rx(7*pi/4) q[5];
cx q[5],q[9];
cx q[9],q[10];
cx q[10],q[12];
cx q[14],q[15];
cx q[11],q[14];
cx q[5],q[11];
cx q[4],q[5];
cx q[3],q[4];
rx(7*pi/4) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/2) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[14],q[8];
cx q[15],q[14];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[10],q[6];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[6];
cx q[9],q[7];
cx q[10],q[9];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
rz(7*pi/4) q[13];
rz(5*pi/4) q[3];
cx q[13],q[9];
cx q[9],q[8];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(3*pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[6],q[4];
cx q[8],q[6];
cx q[9],q[8];
cx q[13],q[9];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[4],q[6];
cx q[2],q[4];
rx(7*pi/4) q[2];
cx q[2],q[4];
cx q[4],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[8],q[13];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[3];
rx(7*pi/4) q[1];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
rx(pi/4) q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[11],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[8],q[6];
cx q[11],q[8];
rx(pi) q[2];
cx q[4],q[5];
cx q[3],q[4];
rx(pi) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[4],q[5];
rx(pi) q[4];
cx q[4],q[5];
cx q[7],q[8];
rx(pi) q[7];
cx q[7],q[8];
rx(pi) q[9];
rx(pi) q[11];
cx q[10],q[11];
cx q[9],q[10];
rx(pi) q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
rx(pi) q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[6],q[8];
cx q[5],q[6];
cx q[3],q[5];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[6],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[10],q[12];
cx q[9],q[10];
cx q[5],q[9];
rx(7*pi/4) q[5];
cx q[5],q[9];
cx q[9],q[10];
cx q[10],q[12];
cx q[14],q[15];
cx q[11],q[14];
cx q[5],q[11];
cx q[4],q[5];
cx q[3],q[4];
rx(7*pi/4) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/2) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[14],q[8];
cx q[15],q[14];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[10],q[6];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[6];
cx q[9],q[7];
cx q[10],q[9];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
rz(7*pi/4) q[13];
rz(5*pi/4) q[3];
cx q[13],q[9];
cx q[9],q[8];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(3*pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[6],q[4];
cx q[8],q[6];
cx q[9],q[8];
cx q[13],q[9];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[4],q[6];
cx q[2],q[4];
rx(7*pi/4) q[2];
cx q[2],q[4];
cx q[4],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[8],q[13];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[3];
rx(7*pi/4) q[1];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
rx(pi/4) q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[11],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[8],q[6];
cx q[11],q[8];
rx(pi) q[2];
cx q[4],q[5];
cx q[3],q[4];
rx(pi) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[4],q[5];
rx(pi) q[4];
cx q[4],q[5];
cx q[7],q[8];
rx(pi) q[7];
cx q[7],q[8];
rx(pi) q[9];
rx(pi) q[11];
cx q[10],q[11];
cx q[9],q[10];
rx(pi) q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
rx(pi) q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[6],q[8];
cx q[5],q[6];
cx q[3],q[5];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[6],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[10],q[12];
cx q[9],q[10];
cx q[5],q[9];
rx(7*pi/4) q[5];
cx q[5],q[9];
cx q[9],q[10];
cx q[10],q[12];
cx q[14],q[15];
cx q[11],q[14];
cx q[5],q[11];
cx q[4],q[5];
cx q[3],q[4];
rx(7*pi/4) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/2) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[14],q[8];
cx q[15],q[14];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[10],q[6];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[6];
cx q[9],q[7];
cx q[10],q[9];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
rz(7*pi/4) q[13];
rz(5*pi/4) q[3];
cx q[13],q[9];
cx q[9],q[8];
cx q[8],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(3*pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[6],q[4];
cx q[8],q[6];
cx q[9],q[8];
cx q[13],q[9];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[4],q[6];
cx q[2],q[4];
rx(7*pi/4) q[2];
cx q[2],q[4];
cx q[4],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[8],q[13];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[3];
rx(7*pi/4) q[1];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[8];
cx q[5],q[7];
cx q[4],q[5];
rx(pi/4) q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[7],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[13];
cx q[14],q[15];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[14];
cx q[14],q[15];
cx q[11],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[8],q[6];
cx q[11],q[8];
cx q[1],q[0];
cx q[7],q[6];
cx q[11],q[10];
cx q[4],q[3];
cx q[10],q[11];
cx q[2],q[1];
cx q[15],q[14];
cx q[8],q[7];
cx q[5],q[4];
cx q[10],q[9];
cx q[12],q[11];
cx q[6],q[5];
