OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1],q[0];
cx q[3],q[2];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[10],q[9];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[14],q[15];
cx q[9],q[14];
rx(3*pi/4) q[9];
cx q[9],q[14];
cx q[14],q[15];
rx(7*pi/4) q[1];
rz(pi/2) q[1];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(3*pi/4) q[3];
cx q[2],q[3];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[5],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
rx(5*pi/4) q[11];
rz(3*pi/2) q[15];
rx(3*pi/2) q[14];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(7*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[10],q[5];
cx q[5],q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[9],q[8];
cx q[0],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[10],q[9];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[14],q[15];
cx q[9],q[14];
rx(3*pi/4) q[9];
cx q[9],q[14];
cx q[14],q[15];
rx(7*pi/4) q[1];
rz(pi/2) q[1];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(3*pi/4) q[3];
cx q[2],q[3];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[5],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
rx(5*pi/4) q[11];
rz(3*pi/2) q[15];
rx(3*pi/2) q[14];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(7*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[10],q[5];
cx q[5],q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[9],q[8];
cx q[0],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[10],q[9];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[14],q[15];
cx q[9],q[14];
rx(3*pi/4) q[9];
cx q[9],q[14];
cx q[14],q[15];
rx(7*pi/4) q[1];
rz(pi/2) q[1];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(3*pi/4) q[3];
cx q[2],q[3];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[5],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
rx(5*pi/4) q[11];
rz(3*pi/2) q[15];
rx(3*pi/2) q[14];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(7*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[10],q[5];
cx q[5],q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[9],q[8];
cx q[0],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[10],q[9];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[14],q[15];
cx q[9],q[14];
rx(3*pi/4) q[9];
cx q[9],q[14];
cx q[14],q[15];
rx(7*pi/4) q[1];
rz(pi/2) q[1];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(3*pi/4) q[3];
cx q[2],q[3];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[5],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
rx(5*pi/4) q[11];
rz(3*pi/2) q[15];
rx(3*pi/2) q[14];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(7*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[10],q[5];
cx q[5],q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[9],q[8];
cx q[0],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[10],q[9];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[14],q[15];
cx q[9],q[14];
rx(3*pi/4) q[9];
cx q[9],q[14];
cx q[14],q[15];
rx(7*pi/4) q[1];
rz(pi/2) q[1];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(3*pi/4) q[3];
cx q[2],q[3];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[5],q[10];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
rx(5*pi/4) q[11];
rz(3*pi/2) q[15];
rx(3*pi/2) q[14];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[5],q[10];
cx q[10],q[5];
cx q[5],q[4];
cx q[4],q[3];
rz(7*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[4];
cx q[10],q[5];
cx q[5],q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[9],q[8];
cx q[0],q[1];
cx q[1],q[0];
