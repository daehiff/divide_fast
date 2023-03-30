OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[4];
cx q[10],q[5];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[15];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
rx(pi/2) q[4];
rz(5*pi/4) q[5];
rx(pi/2) q[6];
rx(pi/2) q[7];
rx(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[12];
rx(pi/2) q[13];
rx(pi/2) q[15];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[4];
cx q[10],q[5];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[15];
cx q[4],q[1];
cx q[6],q[2];
rz(pi/2) q[5];
cx q[13],q[7];
rz(pi/2) q[10];
rz(pi/4) q[11];
rz(15*pi/4) q[2];
rx(pi/2) q[5];
rz(pi/4) q[7];
rx(pi/2) q[10];
rz(pi/2) q[11];
cx q[6],q[2];
rz(pi/2) q[5];
cx q[13],q[7];
rz(pi/2) q[10];
rx(pi/2) q[11];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[15],q[10];
rz(pi/2) q[11];
rz(pi/2) q[13];
rx(pi/2) q[2];
rx(pi/2) q[6];
rx(pi/2) q[7];
rz(7*pi/4) q[10];
rx(pi/2) q[13];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[15],q[10];
rz(pi/2) q[13];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[10];
rx(pi/2) q[13];
rz(pi/2) q[15];
rx(pi/2) q[2];
rx(pi/2) q[6];
rx(pi/2) q[7];
rx(pi/2) q[10];
cx q[14],q[13];
rx(pi/2) q[15];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[10];
rz(pi/2) q[15];
cx q[7],q[1];
rz(pi/2) q[10];
rz(pi/2) q[15];
cx q[9],q[1];
rx(pi/2) q[10];
rx(pi/2) q[15];
rz(11*pi/4) q[1];
rz(pi/2) q[10];
rz(pi/2) q[15];
cx q[4],q[1];
cx q[7],q[1];
rz(pi/2) q[4];
cx q[9],q[1];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[1];
rz(pi/2) q[4];
rx(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[1];
rz(pi/2) q[4];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[1];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[1];
rz(pi/2) q[4];
rx(pi/2) q[7];
rx(pi/2) q[9];
rx(pi/2) q[1];
rz(pi/2) q[7];
rz(11*pi/4) q[9];
rz(pi/2) q[1];
cx q[1],q[3];
cx q[1],q[4];
rx(pi/2) q[1];
cx q[6],q[1];
cx q[10],q[1];
rz(pi/4) q[1];
cx q[3],q[1];
cx q[10],q[1];
cx q[3],q[4];
cx q[12],q[1];
cx q[9],q[4];
cx q[15],q[1];
cx q[10],q[4];
rz(3*pi/4) q[1];
rz(pi/4) q[4];
cx q[3],q[1];
cx q[9],q[4];
cx q[6],q[1];
cx q[10],q[4];
rx(7*pi/2) q[9];
cx q[12],q[1];
cx q[3],q[4];
rz(pi/2) q[6];
rx(pi/2) q[9];
rz(pi/2) q[10];
cx q[15],q[1];
rx(pi/2) q[6];
rx(pi/2) q[10];
rz(pi/2) q[12];
rx(7*pi/2) q[1];
rz(pi/2) q[6];
rz(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[15];
cx q[1],q[4];
rz(pi/2) q[6];
rz(7*pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[15];
cx q[1],q[3];
rz(pi/2) q[4];
rx(pi/2) q[6];
rx(7*pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[15];
rz(pi/2) q[1];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[6];
rz(15*pi/4) q[11];
rx(pi/2) q[1];
rx(pi/2) q[3];
rz(pi/2) q[4];
cx q[12],q[11];
rz(pi/2) q[1];
rz(pi/2) q[3];
cx q[13],q[4];
rz(pi/2) q[12];
rx(pi/2) q[1];
cx q[4],q[3];
rx(pi/2) q[12];
rz(7*pi/4) q[3];
rz(pi/2) q[12];
cx q[4],q[3];
cx q[12],q[9];
cx q[13],q[4];
rz(9*pi/4) q[9];
cx q[12],q[9];
cx q[14],q[13];
rx(7*pi/2) q[9];
rz(pi/2) q[12];
rx(7*pi/2) q[13];
rz(pi/2) q[9];
rx(pi/2) q[12];
rz(pi/2) q[13];
rx(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[13];
cx q[15],q[9];
rx(pi/2) q[12];
cx q[9],q[4];
rz(pi/2) q[12];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[3],q[2];
rz(pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[1];
rx(pi/2) q[2];
cx q[9],q[4];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[4];
cx q[15],q[9];
rz(pi/2) q[2];
rx(pi/2) q[4];
rz(pi/2) q[9];
rx(pi/2) q[2];
rz(pi/2) q[4];
rx(pi/2) q[9];
rz(pi/2) q[2];
rz(pi/2) q[9];
rx(pi/2) q[9];
cx q[9],q[7];
cx q[7],q[4];
cx q[4],q[1];
rz(11*pi/4) q[1];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[7],q[4];
rx(pi/2) q[1];
rz(pi/2) q[4];
cx q[9],q[7];
rz(pi/2) q[1];
rx(pi/2) q[4];
rz(pi/2) q[7];
rx(7*pi/2) q[9];
rz(pi/2) q[4];
rx(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[7];
rz(pi/2) q[9];
cx q[1],q[9];
rx(pi/2) q[7];
rx(pi/2) q[1];
rz(pi/2) q[7];
cx q[2],q[1];
cx q[3],q[1];
cx q[4],q[1];
rz(5*pi/4) q[1];
cx q[1],q[0];
cx q[4],q[0];
cx q[9],q[0];
cx q[14],q[0];
rz(13*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[1];
rz(pi/2) q[2];
cx q[4],q[1];
rx(pi/2) q[2];
cx q[4],q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
cx q[9],q[0];
rz(pi/2) q[2];
rx(pi/2) q[4];
cx q[14],q[0];
cx q[1],q[9];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[9];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[13],q[0];
rz(pi/2) q[9];
cx q[14],q[10];
rz(7*pi/4) q[0];
cx q[10],q[7];
rx(pi/2) q[9];
rz(pi/2) q[14];
cx q[13],q[0];
cx q[7],q[5];
rz(pi/2) q[9];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(15*pi/4) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[0];
cx q[7],q[5];
cx q[14],q[11];
rx(pi/2) q[13];
rz(pi/2) q[0];
cx q[11],q[4];
rz(pi/2) q[5];
cx q[10],q[7];
rz(pi/2) q[13];
rz(pi/4) q[0];
cx q[4],q[1];
rx(pi/2) q[5];
rz(pi/2) q[7];
rx(pi/2) q[13];
cx q[15],q[0];
rz(13*pi/4) q[1];
rz(pi/2) q[5];
rx(pi/2) q[7];
rz(15*pi/4) q[13];
rz(pi/4) q[0];
cx q[4],q[1];
rz(pi/2) q[7];
rx(7*pi/2) q[13];
cx q[15],q[0];
cx q[11],q[4];
rz(pi/2) q[7];
rz(pi/2) q[13];
rx(7*pi/2) q[4];
rx(pi/2) q[7];
cx q[14],q[11];
rx(pi/2) q[13];
rz(pi/2) q[15];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[15];
cx q[1],q[4];
cx q[13],q[7];
rx(pi/2) q[11];
rx(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[1];
cx q[4],q[3];
rz(pi/4) q[7];
rz(pi/2) q[11];
rz(pi/2) q[14];
rx(pi/2) q[1];
cx q[5],q[3];
cx q[13],q[7];
cx q[14],q[10];
rz(pi/4) q[11];
rz(pi/2) q[1];
rz(pi/2) q[7];
rx(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[7];
rz(pi/2) q[10];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[7];
rx(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[14];
cx q[10],q[3];
rz(pi/2) q[7];
cx q[14],q[13];
rz(11*pi/4) q[3];
rx(pi/2) q[7];
cx q[3],q[1];
rz(pi/2) q[7];
cx q[5],q[1];
cx q[8],q[1];
cx q[10],q[1];
cx q[11],q[1];
rz(15*pi/4) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[5],q[3];
cx q[5],q[1];
cx q[8],q[1];
rz(pi/2) q[5];
cx q[10],q[1];
rx(pi/2) q[5];
rz(pi/2) q[8];
cx q[11],q[1];
cx q[10],q[3];
rz(pi/2) q[5];
rx(pi/2) q[8];
rz(pi/2) q[1];
rx(pi/2) q[3];
rz(pi/2) q[8];
rx(7*pi/2) q[10];
rz(pi/2) q[11];
rx(pi/2) q[1];
rz(pi/2) q[8];
rz(pi/2) q[10];
rx(pi/2) q[11];
rz(pi/2) q[1];
rx(pi/2) q[8];
rx(pi/2) q[10];
rz(pi/2) q[11];
cx q[1],q[4];
rz(pi/2) q[8];
rz(pi/2) q[10];
rx(pi/2) q[1];
rx(7*pi/2) q[4];
cx q[4],q[9];
rx(pi/2) q[4];
cx q[4],q[2];
cx q[6],q[2];
cx q[9],q[2];
rz(pi/4) q[2];
cx q[1],q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[9],q[2];
rz(pi/2) q[6];
cx q[10],q[2];
rx(pi/2) q[6];
rz(15*pi/4) q[2];
rz(pi/2) q[6];
cx q[3],q[2];
rz(pi/2) q[6];
cx q[4],q[2];
rx(7*pi/2) q[3];
rx(pi/2) q[6];
cx q[10],q[2];
rz(pi/2) q[3];
rx(7*pi/2) q[4];
rz(pi/2) q[6];
cx q[1],q[2];
rx(pi/2) q[3];
cx q[4],q[9];
rz(pi/2) q[10];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[4];
rx(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[2];
rz(pi/2) q[10];
rz(pi/2) q[2];
cx q[15],q[10];
rz(7*pi/4) q[10];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(13*pi/4) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[3],q[2];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[3];
cx q[6],q[4];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[4];
cx q[9],q[6];
rz(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[6];
cx q[10],q[9];
rx(pi/2) q[2];
rz(pi/2) q[4];
rx(pi/2) q[6];
rx(7*pi/2) q[9];
cx q[12],q[10];
rz(pi/2) q[2];
rz(pi/2) q[6];
cx q[15],q[10];
rz(pi/2) q[12];
rz(pi/2) q[6];
rz(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[15];
rx(pi/2) q[6];
rx(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[15];
rz(pi/2) q[6];
rz(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[15];
rz(pi/2) q[10];
rz(pi/4) q[11];
rx(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[9];
cx q[9],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[2];
cx q[9],q[4];
rz(pi/2) q[2];
rz(pi/2) q[4];
cx q[12],q[9];
rz(pi/2) q[2];
rx(pi/2) q[4];
cx q[9],q[7];
rz(pi/2) q[12];
rx(pi/2) q[2];
rz(pi/2) q[4];
rx(pi/2) q[12];
rz(pi/2) q[2];
cx q[10],q[4];
rz(pi/2) q[12];
cx q[15],q[4];
rz(pi/2) q[12];
rz(5*pi/4) q[4];
rx(pi/2) q[12];
cx q[15],q[4];
rz(pi/2) q[12];
cx q[13],q[4];
cx q[4],q[1];
rz(9*pi/4) q[1];
cx q[4],q[1];
cx q[13],q[4];
cx q[10],q[4];
cx q[14],q[13];
rz(pi/2) q[4];
rz(pi/2) q[10];
rz(pi/2) q[13];
rx(pi/2) q[4];
rx(pi/2) q[10];
rx(pi/2) q[13];
rz(pi/2) q[4];
rz(pi/2) q[10];
rz(pi/2) q[13];
rx(pi/2) q[4];
rz(pi/2) q[10];
cx q[7],q[4];
rx(pi/2) q[10];
cx q[4],q[3];
rz(pi/2) q[10];
cx q[3],q[2];
cx q[2],q[1];
rz(11*pi/4) q[1];
cx q[2],q[1];
rx(pi/2) q[1];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[2];
cx q[7],q[4];
rz(pi/2) q[2];
rx(7*pi/2) q[4];
cx q[9],q[7];
rz(pi/2) q[2];
cx q[9],q[3];
rx(pi/2) q[4];
rz(pi/2) q[7];
rx(pi/2) q[2];
cx q[10],q[4];
rx(pi/2) q[7];
rz(pi/2) q[2];
cx q[4],q[3];
rz(pi/2) q[7];
cx q[3],q[2];
rz(pi/2) q[7];
cx q[2],q[1];
rx(pi/2) q[7];
rz(5*pi/4) q[1];
rz(pi/2) q[7];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[3],q[2];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[2];
cx q[14],q[3];
cx q[10],q[4];
cx q[3],q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[4];
rz(pi/2) q[10];
rz(3*pi/4) q[0];
rz(pi/2) q[2];
rx(pi/2) q[4];
rx(pi/2) q[10];
cx q[3],q[0];
rx(pi/2) q[2];
rz(pi/2) q[10];
rz(pi/2) q[0];
rz(pi/2) q[2];
cx q[14],q[3];
rz(7*pi/2) q[10];
rx(pi/2) q[0];
cx q[9],q[3];
rx(7*pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[9];
rx(pi/2) q[14];
cx q[13],q[0];
rz(pi/2) q[14];
rz(7*pi/4) q[0];
cx q[14],q[10];
cx q[13],q[0];
cx q[10],q[7];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[7],q[5];
rz(pi/2) q[13];
rx(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/4) q[5];
rx(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[7],q[5];
cx q[14],q[11];
rz(pi/2) q[13];
rz(pi/4) q[0];
rz(pi/2) q[5];
cx q[10],q[7];
cx q[11],q[9];
rz(15*pi/4) q[13];
cx q[15],q[0];
cx q[9],q[4];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[13];
rz(pi/4) q[0];
cx q[4],q[2];
rz(pi/2) q[5];
rx(pi/2) q[7];
rx(pi/2) q[13];
cx q[15],q[0];
cx q[2],q[1];
rz(pi/2) q[7];
rz(pi/2) q[13];
rz(13*pi/4) q[1];
rz(pi/2) q[7];
rz(pi/2) q[15];
cx q[2],q[1];
rx(pi/2) q[7];
rx(pi/2) q[15];
rx(7*pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[7];
rz(pi/2) q[15];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[9],q[4];
cx q[13],q[7];
rx(pi/2) q[2];
rx(7*pi/2) q[4];
rz(pi/4) q[7];
cx q[11],q[9];
rz(pi/2) q[2];
cx q[13],q[7];
rx(7*pi/2) q[9];
cx q[14],q[11];
rz(pi/2) q[2];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[2];
rx(pi/2) q[7];
rx(pi/2) q[11];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[2];
rz(pi/2) q[7];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[7];
cx q[14],q[10];
rz(pi/4) q[11];
rx(pi/2) q[13];
cx q[2],q[11];
rx(pi/2) q[7];
rx(pi/2) q[10];
rz(pi/2) q[14];
cx q[2],q[1];
rz(pi/2) q[7];
rz(pi/2) q[10];
rx(pi/2) q[14];
cx q[3],q[1];
cx q[4],q[10];
rz(pi/2) q[14];
rz(pi/2) q[4];
cx q[14],q[13];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[11];
cx q[4],q[1];
cx q[8],q[11];
cx q[5],q[1];
cx q[9],q[11];
cx q[9],q[1];
rz(15*pi/4) q[11];
cx q[10],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[4],q[1];
rx(pi/2) q[3];
cx q[5],q[1];
cx q[9],q[1];
rz(pi/2) q[5];
cx q[10],q[1];
rx(pi/2) q[5];
cx q[9],q[11];
rx(7*pi/2) q[1];
rz(pi/2) q[5];
cx q[8],q[11];
rx(7*pi/2) q[9];
cx q[4],q[11];
rz(pi/2) q[8];
cx q[2],q[11];
rz(pi/2) q[4];
rx(pi/2) q[8];
rz(pi/2) q[2];
rx(pi/2) q[4];
rz(pi/2) q[8];
rz(pi/2) q[11];
rx(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[8];
rx(pi/2) q[11];
rz(pi/2) q[2];
cx q[4],q[10];
rx(pi/2) q[8];
rz(pi/2) q[11];
rz(pi/2) q[2];
rx(pi/2) q[4];
rz(pi/2) q[8];
rz(pi/2) q[10];
cx q[1],q[4];
rx(pi/2) q[2];
rx(pi/2) q[10];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[10];
rx(pi/2) q[1];
cx q[15],q[10];
rz(pi/2) q[1];
rz(7*pi/4) q[10];
cx q[6],q[1];
cx q[12],q[10];
cx q[9],q[1];
rz(15*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(15*pi/4) q[1];
rz(pi/2) q[6];
cx q[2],q[1];
rx(pi/2) q[6];
cx q[3],q[1];
rz(pi/2) q[2];
rz(pi/2) q[6];
cx q[4],q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[3];
rz(pi/2) q[6];
cx q[9],q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[6];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[6];
cx q[10],q[9];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[3];
cx q[9],q[6];
rz(pi/2) q[1];
rz(pi/2) q[2];
cx q[1],q[4];
rx(pi/2) q[1];
rx(7*pi/2) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[6],q[4];
cx q[4],q[3];
rz(13*pi/4) q[3];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[6],q[4];
rx(pi/2) q[3];
rz(pi/2) q[4];
cx q[9],q[6];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[6];
cx q[10],q[9];
rz(pi/2) q[4];
rx(pi/2) q[6];
rx(pi/2) q[9];
cx q[12],q[10];
rz(pi/2) q[4];
rz(pi/2) q[6];
cx q[15],q[10];
rz(pi/2) q[12];
rx(pi/2) q[4];
rz(pi/2) q[6];
rz(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[15];
rz(pi/2) q[4];
rx(pi/2) q[6];
rx(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[15];
rz(pi/2) q[6];
rz(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[15];
rz(pi/2) q[10];
rz(15*pi/4) q[11];
rx(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[10];
rz(pi/2) q[12];
cx q[13],q[10];
rx(pi/2) q[12];
cx q[10],q[9];
rz(pi/2) q[12];
cx q[9],q[4];
cx q[4],q[2];
cx q[2],q[1];
rz(9*pi/4) q[1];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[4],q[2];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[9],q[4];
rx(pi/2) q[2];
rz(pi/2) q[4];
cx q[10],q[9];
rz(pi/2) q[2];
rx(pi/2) q[4];
rx(7*pi/2) q[9];
cx q[13],q[10];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[10];
cx q[14],q[13];
rx(pi/2) q[2];
rx(pi/2) q[4];
rx(pi/2) q[10];
rx(7*pi/2) q[13];
rz(pi/2) q[2];
rz(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[10];
rx(pi/2) q[13];
rx(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[2];
cx q[9],q[4];
rz(pi/2) q[2];
rx(7*pi/2) q[4];
cx q[10],q[9];
rz(pi/2) q[2];
rx(pi/2) q[4];
cx q[12],q[10];
rx(pi/2) q[2];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[2];
rx(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[10];
rz(pi/2) q[12];
cx q[15],q[10];
cx q[10],q[4];
cx q[4],q[3];
cx q[3],q[1];
rz(5*pi/4) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[10],q[4];
rx(7*pi/2) q[4];
cx q[15],q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(11*pi/4) q[1];
cx q[2],q[1];
rz(7*pi/2) q[1];
cx q[3],q[2];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[2];
cx q[7],q[4];
rz(pi/2) q[2];
cx q[9],q[7];
rz(pi/2) q[2];
rz(pi/2) q[7];
cx q[10],q[9];
rx(pi/2) q[2];
rx(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[2];
rz(pi/2) q[7];
rx(pi/2) q[10];
cx q[2],q[9];
rz(pi/2) q[7];
rz(pi/2) q[10];
rx(pi/2) q[7];
rz(7*pi/2) q[9];
rz(7*pi/2) q[10];
cx q[9],q[1];
rz(pi/2) q[7];
rx(7*pi/2) q[10];
rz(5*pi/4) q[1];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cx q[14],q[9];
cx q[9],q[4];
cx q[4],q[0];
rz(13*pi/4) q[0];
cx q[4],q[0];
rz(pi/2) q[0];
cx q[9],q[4];
rx(pi/2) q[0];
rx(pi/2) q[4];
cx q[14],q[9];
rz(pi/2) q[0];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[13],q[0];
rx(pi/2) q[9];
rx(pi/2) q[14];
rz(7*pi/4) q[0];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[13],q[0];
cx q[9],q[1];
cx q[14],q[10];
rz(pi/2) q[0];
rx(pi/2) q[1];
cx q[10],q[7];
rz(pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[1];
cx q[2],q[9];
cx q[7],q[5];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(15*pi/4) q[5];
rz(7*pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/4) q[0];
rx(pi/2) q[2];
cx q[7],q[5];
rz(pi/2) q[9];
cx q[14],q[11];
rx(pi/2) q[13];
cx q[15],q[0];
rz(pi/2) q[2];
rz(pi/2) q[5];
cx q[10],q[7];
rx(pi/2) q[9];
rz(pi/4) q[13];
rz(pi/4) q[0];
rz(pi/2) q[2];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[9];
rx(7*pi/2) q[13];
cx q[15],q[0];
rx(pi/2) q[2];
rz(pi/2) q[5];
rx(pi/2) q[7];
cx q[11],q[9];
rz(pi/2) q[13];
rz(pi/2) q[2];
cx q[9],q[4];
rz(pi/2) q[7];
rx(pi/2) q[13];
rz(pi/2) q[15];
cx q[4],q[2];
rz(pi/2) q[7];
rz(pi/2) q[13];
rx(pi/2) q[15];
cx q[2],q[1];
rx(pi/2) q[7];
rz(pi/2) q[15];
rz(3*pi/4) q[1];
rz(pi/2) q[7];
cx q[2],q[1];
cx q[13],q[7];
rx(7*pi/2) q[1];
cx q[4],q[2];
rz(pi/4) q[7];
rz(pi/2) q[2];
cx q[9],q[4];
cx q[13],q[7];
rx(pi/2) q[2];
rx(7*pi/2) q[4];
rz(pi/2) q[7];
cx q[11],q[9];
rz(pi/2) q[13];
cx q[1],q[4];
rz(pi/2) q[2];
rx(pi/2) q[7];
rz(pi/2) q[9];
cx q[14],q[11];
rx(pi/2) q[13];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[14];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[7];
rz(pi/2) q[11];
rz(pi/2) q[14];
cx q[8],q[1];
rz(pi/2) q[7];
cx q[14],q[10];
rz(pi/4) q[11];
rx(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[10];
rx(pi/2) q[14];
rz(pi/2) q[10];
rz(pi/2) q[14];
rx(pi/2) q[10];
cx q[14],q[13];
rz(pi/2) q[10];
cx q[9],q[10];
cx q[10],q[1];
rx(pi/2) q[9];
cx q[11],q[1];
rz(15*pi/4) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[4],q[1];
cx q[5],q[1];
cx q[8],q[1];
cx q[9],q[1];
rz(pi/2) q[8];
cx q[11],q[1];
rx(pi/2) q[8];
rz(5*pi/4) q[1];
rz(pi/2) q[8];
rz(pi/2) q[11];
cx q[2],q[1];
rz(pi/2) q[8];
rx(pi/2) q[11];
cx q[3],q[1];
rz(pi/2) q[2];
rx(pi/2) q[8];
rz(pi/2) q[11];
cx q[4],q[1];
rx(pi/2) q[2];
rz(pi/2) q[8];
cx q[5],q[1];
rz(pi/2) q[2];
cx q[9],q[1];
rz(pi/2) q[2];
rz(pi/2) q[5];
cx q[10],q[1];
rx(pi/2) q[2];
rx(pi/2) q[5];
rx(7*pi/2) q[9];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[5];
cx q[9],q[10];
rx(pi/2) q[1];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[1];
rx(pi/2) q[10];
cx q[1],q[4];
rz(pi/2) q[10];
cx q[1],q[3];
cx q[4],q[9];
rz(pi/2) q[10];
rz(pi/2) q[1];
cx q[3],q[2];
rx(pi/2) q[4];
rx(pi/2) q[10];
rx(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[10];
rz(pi/2) q[1];
cx q[6],q[2];
rz(pi/4) q[2];
cx q[1],q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[9],q[2];
rz(pi/2) q[6];
cx q[10],q[2];
rx(pi/2) q[6];
rz(15*pi/4) q[2];
rz(pi/2) q[6];
cx q[4],q[2];
rz(pi/2) q[6];
cx q[9],q[2];
rx(7*pi/2) q[4];
rx(pi/2) q[6];
cx q[10],q[2];
cx q[4],q[9];
rz(pi/2) q[6];
cx q[1],q[2];
rx(pi/2) q[4];
rx(7*pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[9];
rz(pi/2) q[10];
cx q[1],q[3];
rz(pi/2) q[2];
rx(pi/2) q[10];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[10];
rz(pi/2) q[2];
cx q[15],q[10];
rz(7*pi/4) q[10];
cx q[12],q[10];
cx q[10],q[9];
cx q[9],q[6];
cx q[6],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(13*pi/4) q[1];
cx q[2],q[1];
rx(7*pi/2) q[1];
cx q[3],q[2];
rz(pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[3];
cx q[6],q[4];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[4];
cx q[9],q[6];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[6];
cx q[10],q[9];
rx(pi/2) q[2];
rx(pi/2) q[4];
rx(pi/2) q[6];
rz(pi/2) q[9];
cx q[12],q[10];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[6];
rx(pi/2) q[9];
cx q[15],q[10];
rz(pi/2) q[12];
rz(pi/2) q[6];
rz(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[15];
rx(pi/2) q[6];
rx(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[15];
rz(pi/2) q[6];
rz(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[15];
rz(pi/2) q[10];
rz(pi/4) q[11];
rx(pi/2) q[10];
cx q[12],q[11];
rz(pi/2) q[10];
rz(pi/2) q[12];
cx q[13],q[10];
rx(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[9];
cx q[9],q[2];
cx q[2],q[1];
rz(7*pi/4) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[9],q[2];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[12],q[9];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[9];
rz(pi/2) q[12];
rz(7*pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[9];
rx(pi/2) q[12];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[12];
rx(pi/2) q[2];
rz(pi/2) q[12];
rz(pi/2) q[2];
rx(pi/2) q[12];
cx q[2],q[9];
rz(pi/2) q[12];
rz(7*pi/2) q[9];
cx q[9],q[1];
cx q[15],q[1];
rz(pi/2) q[9];
rz(11*pi/4) q[1];
rx(pi/2) q[9];
cx q[15],q[1];
rz(pi/2) q[9];
cx q[10],q[9];
cx q[9],q[4];
rz(9*pi/4) q[4];
cx q[9],q[4];
rz(pi/2) q[4];
cx q[10],q[9];
rx(pi/2) q[4];
rz(pi/2) q[9];
cx q[13],q[10];
rz(pi/2) q[4];
rx(pi/2) q[9];
rz(pi/2) q[10];
cx q[14],q[13];
rz(pi/2) q[4];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[13];
cx q[9],q[1];
rx(pi/2) q[4];
rz(pi/2) q[10];
rx(pi/2) q[13];
rx(pi/2) q[1];
rz(pi/2) q[4];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[1];
cx q[2],q[9];
rx(pi/2) q[10];
rz(pi/2) q[2];
rz(7*pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[2];
rx(7*pi/2) q[9];
cx q[14],q[10];
rz(pi/2) q[2];
cx q[9],q[7];
rz(pi/2) q[2];
cx q[7],q[4];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[4],q[2];
rz(11*pi/4) q[2];
cx q[4],q[2];
rz(pi/2) q[2];
cx q[7],q[4];
rx(pi/2) q[2];
rz(pi/2) q[4];
cx q[9],q[7];
rz(pi/2) q[2];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[2];
rz(pi/2) q[4];
rx(pi/2) q[7];
rx(pi/2) q[9];
rx(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[2];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(5*pi/4) q[9];
rz(pi/2) q[4];
rx(pi/2) q[7];
rz(pi/2) q[9];
cx q[10],q[4];
rz(pi/2) q[7];
rx(pi/2) q[9];
cx q[4],q[3];
rz(pi/2) q[9];
cx q[3],q[0];
rz(pi/2) q[9];
rz(3*pi/4) q[0];
rx(pi/2) q[9];
cx q[3],q[0];
rz(pi/2) q[9];
rz(pi/2) q[0];
cx q[4],q[3];
rx(pi/2) q[0];
rz(pi/2) q[3];
cx q[10],q[4];
rz(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[4];
cx q[14],q[10];
cx q[13],q[0];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[10];
rz(pi/2) q[14];
rz(7*pi/4) q[0];
rz(pi/2) q[4];
rx(pi/2) q[10];
rx(pi/2) q[14];
cx q[13],q[0];
rz(pi/2) q[4];
rz(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[4];
cx q[14],q[7];
rz(7*pi/2) q[10];
rz(pi/2) q[13];
rx(pi/2) q[0];
rz(pi/2) q[4];
rx(7*pi/2) q[10];
rx(pi/2) q[13];
rz(pi/2) q[0];
cx q[10],q[5];
rz(pi/2) q[13];
rz(pi/4) q[0];
cx q[7],q[5];
rz(pi/2) q[10];
rz(pi/4) q[13];
cx q[15],q[0];
rz(pi/4) q[5];
rx(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/4) q[0];
cx q[7],q[5];
rz(pi/2) q[10];
rx(pi/2) q[13];
cx q[15],q[0];
cx q[10],q[4];
cx q[14],q[7];
rz(pi/2) q[13];
rz(11*pi/4) q[4];
rz(pi/2) q[7];
rz(pi/2) q[14];
rz(pi/2) q[15];
cx q[10],q[4];
rx(pi/2) q[7];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[4];
rz(pi/2) q[7];
rx(pi/2) q[10];
cx q[14],q[11];
rz(pi/2) q[4];
rx(pi/2) q[7];
rz(pi/2) q[10];
rz(pi/2) q[4];
cx q[10],q[5];
rz(pi/2) q[7];
rx(pi/2) q[4];
rz(pi/2) q[5];
cx q[13],q[7];
rx(pi/2) q[10];
rz(pi/2) q[4];
rx(pi/2) q[5];
rz(pi/4) q[7];
rz(pi/2) q[10];
rz(pi/2) q[5];
cx q[13],q[7];
rz(7*pi/2) q[10];
rz(pi/2) q[5];
rz(pi/2) q[7];
rx(7*pi/2) q[10];
rz(pi/2) q[13];
rx(pi/2) q[5];
rx(pi/2) q[7];
rz(pi/2) q[10];
rx(pi/2) q[13];
cx q[4],q[10];
rz(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[13];
rz(pi/2) q[7];
rz(7*pi/2) q[10];
rx(pi/2) q[13];
cx q[10],q[3];
rx(pi/2) q[7];
cx q[9],q[3];
rz(pi/2) q[7];
rz(pi/2) q[10];
rz(15*pi/4) q[3];
rx(pi/2) q[10];
cx q[9],q[3];
rz(pi/2) q[10];
rz(pi/2) q[9];
cx q[11],q[10];
rx(pi/2) q[9];
rz(13*pi/4) q[10];
rz(pi/2) q[9];
cx q[11],q[10];
rz(pi/2) q[9];
rz(pi/2) q[10];
cx q[14],q[11];
rx(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[11];
rz(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[11];
cx q[10],q[3];
rz(pi/2) q[11];
rz(pi/2) q[3];
rz(pi/2) q[10];
rz(pi/4) q[11];
rx(pi/2) q[3];
cx q[4],q[10];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(7*pi/2) q[10];
rx(pi/2) q[4];
rx(pi/2) q[10];
rz(pi/2) q[4];
rz(pi/2) q[10];
rx(pi/2) q[4];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cx q[11],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[3],q[2];
rx(pi/2) q[1];
rz(pi/2) q[2];
cx q[4],q[3];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[3];
cx q[8],q[4];
rz(pi/2) q[2];
rx(pi/2) q[3];
rx(7*pi/2) q[4];
cx q[9],q[8];
rx(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[8];
cx q[10],q[9];
rz(pi/2) q[2];
rx(pi/2) q[8];
rz(pi/2) q[9];
cx q[11],q[10];
rz(pi/2) q[2];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[11];
rx(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[11];
rz(pi/2) q[2];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[11];
rz(pi/2) q[2];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi) q[2];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[10];
cx q[10],q[6];
cx q[6],q[1];
rz(15*pi/4) q[1];
cx q[6],q[1];
rz(pi/2) q[1];
cx q[10],q[6];
rx(pi/2) q[1];
rz(pi/2) q[6];
rz(pi/2) q[10];
rz(pi/2) q[1];
rx(pi/2) q[6];
rx(pi/2) q[10];
cx q[1],q[0];
rz(pi/2) q[6];
rz(pi/2) q[10];
rz(pi/2) q[6];
rz(pi/2) q[10];
rx(pi/2) q[6];
rx(pi/2) q[10];
rz(pi/2) q[6];
rz(pi/2) q[10];
cx q[15],q[10];
rz(7*pi/4) q[10];
cx q[15],q[10];
rz(pi/2) q[10];
cx q[15],q[12];
cx q[12],q[6];
rx(pi/2) q[10];
cx q[6],q[3];
rz(pi/2) q[10];
rz(13*pi/4) q[3];
rz(pi/2) q[10];
cx q[6],q[3];
rx(pi/2) q[10];
rz(pi/2) q[3];
cx q[12],q[6];
rz(pi/2) q[10];
rx(pi/2) q[3];
rz(pi/2) q[6];
cx q[15],q[12];
rz(pi/2) q[3];
rx(pi/2) q[6];
rz(pi/2) q[12];
rz(pi/2) q[15];
cx q[4],q[3];
rz(pi/2) q[6];
rx(pi/2) q[12];
rx(pi/2) q[15];
cx q[4],q[0];
cx q[10],q[3];
rx(pi/2) q[6];
rz(pi/2) q[12];
rz(pi/2) q[15];
cx q[10],q[0];
cx q[13],q[3];
rz(pi/2) q[6];
cx q[12],q[11];
cx q[14],q[3];
rz(pi/2) q[6];
rz(15*pi/4) q[11];
cx q[14],q[0];
rz(9*pi/4) q[3];
rx(pi/2) q[6];
cx q[12],q[11];
rz(13*pi/4) q[0];
cx q[4],q[3];
rz(pi/2) q[6];
rz(pi/2) q[12];
cx q[1],q[0];
rz(pi/2) q[6];
rx(pi/2) q[12];
cx q[4],q[0];
rz(pi/2) q[1];
rz(pi/2) q[12];
cx q[10],q[0];
rx(pi/2) q[1];
rx(7*pi/2) q[4];
cx q[12],q[9];
rz(pi/2) q[1];
cx q[10],q[3];
rx(pi/2) q[4];
rz(7*pi/4) q[9];
cx q[13],q[3];
rz(pi/2) q[4];
cx q[12],q[9];
rz(pi/2) q[10];
cx q[14],q[3];
rz(pi/2) q[4];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[12];
rx(7*pi/2) q[13];
cx q[14],q[0];
rx(pi/2) q[4];
rx(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[4];
cx q[15],q[9];
rx(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[0];
rx(pi) q[4];
rz(5*pi/4) q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[14];
cx q[13],q[0];
cx q[15],q[9];
cx q[14],q[10];
rz(pi/2) q[12];
rz(7*pi/4) q[0];
cx q[10],q[7];
rx(pi/2) q[9];
rx(pi/2) q[12];
rz(pi/2) q[14];
cx q[13],q[0];
cx q[7],q[5];
rz(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(15*pi/4) q[5];
rz(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[0];
cx q[7],q[5];
rx(pi/2) q[9];
cx q[14],q[11];
rx(pi/2) q[13];
rz(pi/2) q[0];
cx q[11],q[3];
rz(pi/2) q[5];
cx q[10],q[7];
rz(pi/2) q[9];
rz(pi/2) q[13];
cx q[15],q[0];
rz(13*pi/4) q[3];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[13];
rz(pi/4) q[0];
cx q[11],q[3];
rz(pi/2) q[5];
rx(pi/2) q[7];
rz(15*pi/4) q[13];
cx q[15],q[0];
rx(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[7];
cx q[14],q[11];
rx(7*pi/2) q[13];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[7];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[7];
rx(pi/2) q[11];
rz(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[11];
rx(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[7];
cx q[11],q[8];
cx q[14],q[10];
rz(pi/2) q[13];
rx(pi/2) q[15];
rz(pi/2) q[0];
cx q[8],q[1];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[10];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[0];
rz(15*pi/4) q[1];
rz(pi) q[3];
rz(pi/2) q[7];
rx(pi/2) q[10];
rx(pi/2) q[14];
rz(pi/2) q[15];
cx q[8],q[1];
rx(pi) q[3];
rz(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[1];
cx q[11],q[8];
rx(pi/2) q[10];
rx(pi/2) q[14];
rx(pi/2) q[1];
rz(pi/2) q[8];
rz(pi/2) q[10];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[1];
rx(pi/2) q[8];
rz(pi/2) q[10];
rx(pi/2) q[11];
rz(pi/2) q[14];
rx(pi/2) q[1];
rz(pi/2) q[8];
rx(pi/2) q[10];
rz(pi/2) q[11];
rx(pi/2) q[14];
rz(pi/2) q[1];
rx(pi/2) q[8];
rz(pi/2) q[10];
rx(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[1];
rz(pi/2) q[8];
rz(pi/2) q[10];
rz(pi/2) q[11];
rz(pi/2) q[14];
rx(pi/2) q[1];
rz(pi/2) q[8];
rx(pi) q[10];
rz(pi/2) q[11];
rz(pi) q[14];
rz(pi/2) q[1];
rx(pi/2) q[8];
rx(pi/2) q[11];
rz(pi/2) q[1];
rz(pi/2) q[8];
rz(pi/2) q[11];
rz(pi) q[1];
rz(pi/2) q[8];
rz(pi/2) q[11];
rx(pi) q[11];
