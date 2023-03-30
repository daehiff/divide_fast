OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[13],q[14];
cx q[10],q[11];
cx q[2],q[1];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[10],q[9];
cx q[4],q[3];
cx q[14],q[13];
cx q[6],q[7];
cx q[1],q[0];
cx q[1],q[2];
cx q[10],q[11];
cx q[15],q[14];
cx q[6],q[5];
cx q[8],q[7];
cx q[12],q[13];
rz(pi) q[4];
rz(pi) q[9];
cx q[12],q[10];
rz(pi) q[10];
cx q[12],q[10];
rx(pi) q[3];
rx(pi) q[9];
cx q[13],q[14];
cx q[12],q[13];
rx(pi) q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[13],q[14];
rx(pi) q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[6],q[2];
cx q[10],q[6];
cx q[11],q[10];
cx q[7],q[6];
cx q[6],q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[7],q[6];
cx q[11],q[10];
cx q[10],q[1];
rz(3*pi/2) q[1];
cx q[10],q[1];
cx q[11],q[10];
cx q[6],q[5];
rz(pi/4) q[5];
cx q[6],q[5];
rz(pi/2) q[10];
rz(pi/4) q[1];
cx q[4],q[8];
rx(3*pi/2) q[4];
cx q[4],q[8];
cx q[6],q[9];
cx q[5],q[6];
rx(7*pi/4) q[5];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[2],q[7];
cx q[0],q[2];
rx(7*pi/4) q[0];
cx q[0],q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[9];
rx(3*pi/2) q[0];
cx q[14],q[15];
cx q[13],q[14];
cx q[2],q[13];
rx(pi/2) q[2];
cx q[2],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[8],q[12];
rx(7*pi/4) q[8];
cx q[8],q[12];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
rx(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5*pi/4) q[9];
rz(7*pi/4) q[7];
cx q[9],q[1];
rz(7*pi/4) q[1];
cx q[9],q[1];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[5],q[2];
cx q[6],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
rz(pi/2) q[2];
cx q[5],q[2];
cx q[6],q[5];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[9],q[7];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
rz(pi) q[4];
rz(pi) q[9];
cx q[12],q[10];
rz(pi) q[10];
cx q[12],q[10];
rx(pi) q[3];
rx(pi) q[9];
cx q[13],q[14];
cx q[12],q[13];
rx(pi) q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[13],q[14];
rx(pi) q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[6],q[2];
cx q[10],q[6];
cx q[11],q[10];
cx q[7],q[6];
cx q[6],q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[7],q[6];
cx q[11],q[10];
cx q[10],q[1];
rz(3*pi/2) q[1];
cx q[10],q[1];
cx q[11],q[10];
cx q[6],q[5];
rz(pi/4) q[5];
cx q[6],q[5];
rz(pi/2) q[10];
rz(pi/4) q[1];
cx q[4],q[8];
rx(3*pi/2) q[4];
cx q[4],q[8];
cx q[6],q[9];
cx q[5],q[6];
rx(7*pi/4) q[5];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[2],q[7];
cx q[0],q[2];
rx(7*pi/4) q[0];
cx q[0],q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[9];
rx(3*pi/2) q[0];
cx q[14],q[15];
cx q[13],q[14];
cx q[2],q[13];
rx(pi/2) q[2];
cx q[2],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[8],q[12];
rx(7*pi/4) q[8];
cx q[8],q[12];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
rx(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5*pi/4) q[9];
rz(7*pi/4) q[7];
cx q[9],q[1];
rz(7*pi/4) q[1];
cx q[9],q[1];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[5],q[2];
cx q[6],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
rz(pi/2) q[2];
cx q[5],q[2];
cx q[6],q[5];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[9],q[7];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
rz(pi) q[4];
rz(pi) q[9];
cx q[12],q[10];
rz(pi) q[10];
cx q[12],q[10];
rx(pi) q[3];
rx(pi) q[9];
cx q[13],q[14];
cx q[12],q[13];
rx(pi) q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[13],q[14];
rx(pi) q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[6],q[2];
cx q[10],q[6];
cx q[11],q[10];
cx q[7],q[6];
cx q[6],q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[7],q[6];
cx q[11],q[10];
cx q[10],q[1];
rz(3*pi/2) q[1];
cx q[10],q[1];
cx q[11],q[10];
cx q[6],q[5];
rz(pi/4) q[5];
cx q[6],q[5];
rz(pi/2) q[10];
rz(pi/4) q[1];
cx q[4],q[8];
rx(3*pi/2) q[4];
cx q[4],q[8];
cx q[6],q[9];
cx q[5],q[6];
rx(7*pi/4) q[5];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[2],q[7];
cx q[0],q[2];
rx(7*pi/4) q[0];
cx q[0],q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[9];
rx(3*pi/2) q[0];
cx q[14],q[15];
cx q[13],q[14];
cx q[2],q[13];
rx(pi/2) q[2];
cx q[2],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[8],q[12];
rx(7*pi/4) q[8];
cx q[8],q[12];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
rx(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5*pi/4) q[9];
rz(7*pi/4) q[7];
cx q[9],q[1];
rz(7*pi/4) q[1];
cx q[9],q[1];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[5],q[2];
cx q[6],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
rz(pi/2) q[2];
cx q[5],q[2];
cx q[6],q[5];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[9],q[7];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
rz(pi) q[4];
rz(pi) q[9];
cx q[12],q[10];
rz(pi) q[10];
cx q[12],q[10];
rx(pi) q[3];
rx(pi) q[9];
cx q[13],q[14];
cx q[12],q[13];
rx(pi) q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[13],q[14];
rx(pi) q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[6],q[2];
cx q[10],q[6];
cx q[11],q[10];
cx q[7],q[6];
cx q[6],q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[7],q[6];
cx q[11],q[10];
cx q[10],q[1];
rz(3*pi/2) q[1];
cx q[10],q[1];
cx q[11],q[10];
cx q[6],q[5];
rz(pi/4) q[5];
cx q[6],q[5];
rz(pi/2) q[10];
rz(pi/4) q[1];
cx q[4],q[8];
rx(3*pi/2) q[4];
cx q[4],q[8];
cx q[6],q[9];
cx q[5],q[6];
rx(7*pi/4) q[5];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[2],q[7];
cx q[0],q[2];
rx(7*pi/4) q[0];
cx q[0],q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[9];
rx(3*pi/2) q[0];
cx q[14],q[15];
cx q[13],q[14];
cx q[2],q[13];
rx(pi/2) q[2];
cx q[2],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[8],q[12];
rx(7*pi/4) q[8];
cx q[8],q[12];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
rx(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5*pi/4) q[9];
rz(7*pi/4) q[7];
cx q[9],q[1];
rz(7*pi/4) q[1];
cx q[9],q[1];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[5],q[2];
cx q[6],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
rz(pi/2) q[2];
cx q[5],q[2];
cx q[6],q[5];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[9],q[7];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
rz(pi) q[4];
rz(pi) q[9];
cx q[12],q[10];
rz(pi) q[10];
cx q[12],q[10];
rx(pi) q[3];
rx(pi) q[9];
cx q[13],q[14];
cx q[12],q[13];
rx(pi) q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[13],q[14];
rx(pi) q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[11],q[10];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[6],q[2];
cx q[10],q[6];
cx q[11],q[10];
cx q[7],q[6];
cx q[6],q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[7],q[6];
cx q[11],q[10];
cx q[10],q[1];
rz(3*pi/2) q[1];
cx q[10],q[1];
cx q[11],q[10];
cx q[6],q[5];
rz(pi/4) q[5];
cx q[6],q[5];
rz(pi/2) q[10];
rz(pi/4) q[1];
cx q[4],q[8];
rx(3*pi/2) q[4];
cx q[4],q[8];
cx q[6],q[9];
cx q[5],q[6];
rx(7*pi/4) q[5];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[2],q[7];
cx q[0],q[2];
rx(7*pi/4) q[0];
cx q[0],q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[9];
rx(3*pi/2) q[0];
cx q[14],q[15];
cx q[13],q[14];
cx q[2],q[13];
rx(pi/2) q[2];
cx q[2],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[8],q[12];
rx(7*pi/4) q[8];
cx q[8],q[12];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
rx(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(5*pi/4) q[9];
rz(7*pi/4) q[7];
cx q[9],q[1];
rz(7*pi/4) q[1];
cx q[9],q[1];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
cx q[5],q[2];
cx q[6],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[2];
rz(pi/2) q[2];
cx q[5],q[2];
cx q[6],q[5];
cx q[9],q[6];
rz(pi/4) q[6];
cx q[9],q[6];
cx q[2],q[3];
rx(5*pi/4) q[2];
cx q[2],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[9],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[9],q[7];
cx q[4],q[3];
rz(3*pi/4) q[3];
cx q[4],q[3];
cx q[1],q[2];
cx q[10],q[11];
cx q[15],q[14];
cx q[6],q[5];
cx q[8],q[7];
cx q[12],q[13];
cx q[10],q[9];
cx q[4],q[3];
cx q[14],q[13];
cx q[6],q[7];
cx q[1],q[0];
cx q[13],q[14];
cx q[10],q[11];
cx q[2],q[1];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
