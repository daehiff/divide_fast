OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[10],q[12];
cx q[20],q[23];
cx q[22],q[24];
cx q[12],q[10];
cx q[23],q[20];
cx q[24],q[22];
cx q[10],q[12];
cx q[20],q[23];
cx q[22],q[24];
cx q[3],q[10];
rz(pi/2) q[12];
cx q[16],q[22];
cx q[17],q[20];
cx q[18],q[23];
cx q[21],q[24];
cx q[10],q[3];
rx(pi/2) q[12];
cx q[22],q[16];
cx q[20],q[17];
cx q[23],q[18];
cx q[24],q[21];
cx q[3],q[10];
rz(pi/2) q[12];
cx q[16],q[22];
cx q[17],q[20];
cx q[18],q[23];
cx q[21],q[24];
rz(pi/2) q[3];
cx q[6],q[22];
cx q[9],q[18];
rz(pi/2) q[10];
cx q[13],q[23];
cx q[14],q[16];
cx q[15],q[21];
rz(pi/2) q[17];
cx q[19],q[24];
rx(pi/2) q[3];
cx q[22],q[6];
cx q[18],q[9];
rx(pi/2) q[10];
cx q[23],q[13];
cx q[16],q[14];
cx q[21],q[15];
rx(pi/2) q[17];
cx q[24],q[19];
rz(pi/2) q[3];
cx q[6],q[22];
cx q[9],q[18];
rz(pi/2) q[10];
cx q[13],q[23];
cx q[14],q[16];
cx q[15],q[21];
rz(pi/2) q[17];
cx q[19],q[24];
cx q[1],q[9];
rz(pi/2) q[3];
rz(pi/2) q[6];
cx q[7],q[14];
cx q[8],q[23];
cx q[11],q[15];
rz(pi/2) q[13];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[9],q[1];
rx(pi/2) q[3];
rx(pi/2) q[6];
cx q[14],q[7];
cx q[23],q[8];
cx q[15],q[11];
rx(pi/2) q[13];
rx(pi/2) q[16];
rx(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[24];
cx q[1],q[9];
rz(pi/2) q[3];
rz(pi/2) q[6];
cx q[7],q[14];
cx q[8],q[23];
cx q[11],q[15];
rz(pi/2) q[13];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[24];
rz(pi/2) q[1];
cx q[2],q[11];
cx q[5],q[7];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[1];
cx q[11],q[2];
cx q[7],q[5];
rx(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[14];
rx(pi/2) q[15];
rx(pi/2) q[18];
rx(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[1];
cx q[2],q[11];
cx q[5],q[7];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[2];
cx q[4],q[7];
rz(pi/2) q[5];
rz(pi/2) q[9];
rz(pi/2) q[11];
cx q[19],q[24];
cx q[20],q[22];
rz(pi/2) q[23];
rx(pi/2) q[2];
cx q[7],q[4];
rx(pi/2) q[5];
rx(pi/2) q[9];
rx(pi/2) q[11];
rz(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[2];
cx q[4],q[7];
rz(pi/2) q[5];
rz(pi/2) q[9];
rz(pi/2) q[11];
rx(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
cx q[0],q[4];
rz(pi/2) q[7];
cx q[19],q[23];
rz(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[4],q[0];
rx(pi/2) q[7];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[0],q[4];
rz(pi/2) q[7];
rx(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rx(pi/2) q[4];
cx q[13],q[24];
cx q[19],q[22];
rz(pi/2) q[23];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[19];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[19];
rx(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
cx q[17],q[23];
rz(pi/2) q[19];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[12],q[19];
cx q[17],q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[17],q[18];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[17];
rz(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[10],q[24];
cx q[16],q[22];
rx(pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[23];
cx q[10],q[21];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
cx q[15],q[23];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[10],q[19];
rz(pi/2) q[15];
cx q[16],q[20];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[10],q[17];
rx(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[10];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[17];
rx(pi/2) q[19];
rx(pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[0],q[24];
cx q[6],q[15];
rx(pi/2) q[10];
cx q[13],q[22];
rz(pi/2) q[16];
rx(pi/2) q[17];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[6];
cx q[8],q[21];
rz(pi/2) q[10];
cx q[13],q[18];
rz(pi/2) q[15];
rz(pi/2) q[17];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[6];
cx q[13],q[16];
rx(pi/2) q[15];
rz(pi/2) q[17];
rz(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[6];
cx q[14],q[23];
rz(pi/2) q[15];
rz(pi/2) q[16];
rx(pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[5],q[19];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[5],q[6];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[6];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[16];
rx(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[6];
cx q[8],q[14];
rx(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[6];
cx q[7],q[21];
rz(pi/2) q[8];
cx q[12],q[18];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[19];
rx(pi/2) q[23];
rz(pi/2) q[6];
cx q[7],q[17];
rx(pi/2) q[8];
rz(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[0],q[23];
rx(pi/2) q[6];
cx q[7],q[10];
rz(pi/2) q[8];
rx(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[6];
cx q[7],q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[4],q[6];
cx q[7],q[8];
rz(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rx(pi/2) q[23];
cx q[13],q[4];
rz(pi/2) q[6];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[17];
rx(pi/2) q[18];
rx(pi/2) q[21];
rz(pi/2) q[23];
cx q[4],q[14];
rx(pi/2) q[6];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[2],q[21];
rz(pi/2) q[6];
rz(pi/2) q[8];
rx(pi/2) q[10];
rz(pi/2) q[14];
rz(pi/2) q[17];
rx(pi/2) q[23];
cx q[1],q[17];
cx q[2],q[16];
rz(pi/2) q[8];
rz(pi/2) q[10];
rx(pi/2) q[14];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[1],q[9];
cx q[2],q[12];
rx(pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[17];
rx(pi/2) q[21];
rz(pi/2) q[8];
cx q[9],q[19];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[16];
rx(pi/2) q[17];
rz(pi/2) q[21];
cx q[0],q[8];
cx q[9],q[18];
rx(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[19];
cx q[0],q[3];
cx q[16],q[5];
rz(pi/2) q[8];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[3];
rx(pi/2) q[8];
cx q[9],q[14];
rz(pi/2) q[12];
rx(pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[19];
cx q[1],q[9];
rx(pi/2) q[3];
rz(pi/2) q[8];
rx(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[19];
cx q[18],q[0];
cx q[1],q[6];
rz(pi/2) q[3];
cx q[8],q[22];
rz(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[19];
cx q[0],q[15];
cx q[21],q[3];
cx q[4],q[12];
rx(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[19];
cx q[22],q[23];
cx q[4],q[11];
rz(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[15];
cx q[22],q[17];
rz(pi/2) q[23];
cx q[13],q[4];
rz(pi/2) q[11];
rx(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[17];
rx(pi/2) q[23];
rz(pi/2) q[4];
rx(pi/2) q[11];
rz(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[17];
rz(pi/2) q[23];
cx q[0],q[14];
rx(pi/2) q[4];
cx q[7],q[15];
rz(pi/2) q[11];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[17];
rz(pi/2) q[23];
cx q[0],q[10];
cx q[2],q[11];
rz(pi/2) q[4];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[14];
cx q[15],q[24];
rz(pi/2) q[17];
rx(pi/2) q[23];
cx q[0],q[4];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[17];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[18],q[0];
rz(pi/2) q[4];
cx q[5],q[23];
rx(pi/2) q[10];
cx q[22],q[12];
rz(pi/2) q[14];
rz(pi/2) q[17];
rx(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[4];
cx q[22],q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[18];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[4];
cx q[8],q[22];
rz(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[18];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[4];
cx q[5],q[14];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[4];
cx q[6],q[24];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[14];
cx q[15],q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[5],q[12];
cx q[6],q[23];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[15],q[20];
rz(pi/2) q[22];
rx(pi/2) q[24];
cx q[5],q[9];
rz(pi/2) q[12];
rz(pi/2) q[14];
cx q[15],q[19];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[5],q[8];
rz(pi/2) q[9];
rx(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
cx q[5],q[0];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[16],q[5];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[5];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[12];
rz(pi/2) q[16];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[3],q[22];
rx(pi/2) q[5];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[23];
rz(pi/2) q[0];
cx q[3],q[20];
rz(pi/2) q[5];
rx(pi/2) q[8];
rz(pi/2) q[9];
cx q[11],q[23];
cx q[15],q[12];
rz(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[22];
rx(pi/2) q[0];
cx q[3],q[18];
cx q[15],q[5];
rz(pi/2) q[8];
rz(pi/2) q[12];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[0];
cx q[3],q[16];
cx q[15],q[4];
rz(pi/2) q[5];
rx(pi/2) q[12];
rz(pi/2) q[18];
rx(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[23];
cx q[3],q[14];
rz(pi/2) q[4];
rx(pi/2) q[5];
cx q[7],q[15];
rz(pi/2) q[12];
rz(pi/2) q[16];
rx(pi/2) q[18];
rz(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[3],q[9];
rx(pi/2) q[4];
rz(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(3*pi/2) q[23];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[18];
rx(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[4];
rx(pi/2) q[5];
cx q[6],q[22];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(3*pi/2) q[16];
rx(pi/2) q[18];
rz(pi/2) q[20];
rx(pi/2) q[23];
rx(pi/2) q[4];
rz(pi/2) q[5];
cx q[6],q[13];
rz(pi/2) q[9];
rz(3*pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[4];
cx q[6],q[12];
rz(pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[16];
rx(pi/2) q[22];
cx q[3],q[4];
cx q[6],q[7];
rx(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[22];
cx q[21],q[3];
rz(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[22];
rz(pi/2) q[3];
rx(pi/2) q[4];
rx(pi/2) q[7];
rz(pi/2) q[12];
rz(pi) q[13];
rz(pi) q[21];
rx(pi/2) q[22];
rx(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[7];
rz(pi) q[12];
rz(pi/2) q[13];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[3];
rz(pi) q[4];
rz(3*pi/2) q[7];
cx q[11],q[22];
rz(pi/2) q[12];
rx(pi/2) q[13];
rx(pi/2) q[21];
cx q[6],q[3];
rz(pi/2) q[4];
rz(pi/2) q[7];
cx q[11],q[20];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[21];
rz(pi/2) q[22];
cx q[6],q[0];
rz(pi/2) q[3];
rx(pi/2) q[4];
rx(pi/2) q[7];
cx q[11],q[19];
rz(pi/2) q[12];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[0];
cx q[1],q[6];
rx(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[7];
cx q[11],q[18];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[0];
rz(3*pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[6];
cx q[11],q[15];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[6];
cx q[11],q[10];
rz(pi/2) q[15];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi) q[20];
rx(pi/2) q[22];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[3];
rx(pi/2) q[6];
cx q[11],q[9];
rz(pi/2) q[10];
rx(pi/2) q[15];
rz(pi/2) q[18];
rz(3*pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[6];
cx q[11],q[8];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[15];
rz(3*pi/2) q[18];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[0];
cx q[11],q[5];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[15];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[20];
cx q[11],q[3];
rz(pi/2) q[5];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[15];
rx(pi/2) q[18];
rz(pi/2) q[19];
cx q[11],q[0];
rz(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[8];
rz(pi) q[9];
rz(pi/2) q[10];
rx(pi/2) q[15];
rz(pi/2) q[18];
rz(pi/2) q[0];
cx q[2],q[11];
rx(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[10];
rz(pi/2) q[15];
rx(pi/2) q[0];
rz(3*pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi) q[11];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[8];
rx(pi/2) q[11];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[11];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[1];
cx q[1],q[4];
cx q[4],q[5];
cx q[2],q[5];
cx q[2],q[10];
cx q[5],q[6];
cx q[2],q[15];
cx q[6],q[7];
cx q[0],q[6];
cx q[7],q[8];
cx q[1],q[7];
cx q[8],q[12];
cx q[1],q[10];
cx q[4],q[7];
cx q[8],q[13];
cx q[1],q[9];
cx q[4],q[12];
cx q[10],q[11];
cx q[0],q[10];
cx q[3],q[11];
cx q[7],q[9];
cx q[3],q[5];
cx q[4],q[7];
cx q[6],q[10];
cx q[9],q[11];
cx q[5],q[9];
cx q[11],q[12];
cx q[12],q[14];
cx q[11],q[14];
cx q[12],q[13];
cx q[0],q[12];
cx q[6],q[11];
cx q[10],q[13];
cx q[0],q[20];
cx q[10],q[17];
cx q[13],q[15];
cx q[5],q[20];
cx q[12],q[15];
cx q[15],q[17];
cx q[4],q[15];
cx q[12],q[17];
cx q[11],q[17];
cx q[12],q[16];
cx q[2],q[16];
cx q[7],q[12];
cx q[17],q[18];
cx q[1],q[16];
cx q[2],q[14];
cx q[7],q[18];
cx q[17],q[20];
cx q[1],q[8];
cx q[3],q[14];
cx q[10],q[16];
cx q[11],q[17];
cx q[12],q[18];
cx q[3],q[10];
cx q[6],q[16];
cx q[7],q[8];
cx q[13],q[14];
cx q[0],q[13];
cx q[8],q[15];
cx q[16],q[19];
cx q[0],q[23];
cx q[5],q[13];
cx q[6],q[16];
cx q[18],q[19];
cx q[6],q[21];
cx q[12],q[19];
cx q[13],q[17];
cx q[15],q[16];
cx q[19],q[20];
cx q[9],q[19];
cx q[12],q[20];
cx q[9],q[14];
cx q[20],q[21];
cx q[1],q[20];
cx q[9],q[10];
cx q[19],q[21];
cx q[1],q[18];
cx q[5],q[20];
cx q[3],q[18];
cx q[7],q[20];
cx q[3],q[23];
cx q[10],q[20];
cx q[14],q[18];
cx q[4],q[23];
cx q[16],q[20];
cx q[4],q[12];
cx q[6],q[23];
cx q[20],q[24];
cx q[10],q[12];
cx q[13],q[23];
cx q[19],q[23];
cx q[18],q[23];
cx q[23],q[22];
cx q[21],q[22];
cx q[19],q[22];
cx q[18],q[22];
cx q[17],q[18];
cx q[22],q[23];
cx q[18],q[24];
cx q[23],q[21];
cx q[0],q[24];
cx q[22],q[21];
cx q[1],q[24];
cx q[9],q[21];
cx q[3],q[24];
cx q[14],q[21];
cx q[4],q[24];
cx q[23],q[14];
cx q[15],q[21];
cx q[5],q[24];
cx q[21],q[20];
cx q[7],q[24];
cx q[20],q[19];
cx q[21],q[22];
cx q[8],q[19];
cx q[22],q[14];
cx q[20],q[21];
cx q[8],q[24];
cx q[12],q[19];
cx q[21],q[18];
cx q[10],q[24];
cx q[23],q[12];
cx q[21],q[14];
cx q[19],q[20];
cx q[11],q[24];
cx q[21],q[12];
cx q[20],q[18];
cx q[24],q[16];
cx q[19],q[18];
cx q[24],q[15];
cx q[18],q[19];
cx q[18],q[15];
cx q[19],q[17];
cx q[24],q[20];
cx q[19],q[12];
cx q[18],q[14];
cx q[17],q[16];
cx q[23],q[20];
cx q[23],q[13];
cx q[16],q[17];
cx q[22],q[20];
cx q[20],q[11];
cx q[16],q[15];
cx q[20],q[10];
cx q[17],q[11];
cx q[15],q[12];
cx q[23],q[16];
cx q[20],q[9];
cx q[14],q[12];
cx q[22],q[16];
cx q[12],q[11];
cx q[14],q[18];
cx q[21],q[16];
cx q[18],q[8];
cx q[11],q[12];
cx q[20],q[16];
cx q[12],q[10];
cx q[18],q[13];
cx q[11],q[10];
cx q[17],q[13];
cx q[10],q[11];
cx q[14],q[13];
cx q[11],q[9];
cx q[20],q[14];
cx q[9],q[11];
cx q[24],q[9];
cx q[16],q[9];
cx q[24],q[23];
cx q[15],q[9];
cx q[23],q[12];
cx q[15],q[7];
cx q[10],q[9];
cx q[22],q[12];
cx q[15],q[6];
cx q[9],q[8];
cx q[18],q[12];
cx q[18],q[6];
cx q[8],q[9];
cx q[13],q[12];
cx q[23],q[15];
cx q[12],q[7];
cx q[16],q[15];
cx q[15],q[5];
cx q[9],q[7];
cx q[22],q[12];
cx q[14],q[5];
cx q[7],q[9];
cx q[14],q[3];
cx q[10],q[5];
cx q[24],q[7];
cx q[24],q[1];
cx q[13],q[3];
cx q[19],q[7];
cx q[14],q[9];
cx q[19],q[4];
cx q[8],q[7];
cx q[16],q[14];
cx q[8],q[1];
cx q[15],q[4];
cx q[7],q[5];
cx q[19],q[11];
cx q[10],q[4];
cx q[5],q[7];
cx q[11],q[6];
cx q[23],q[15];
cx q[7],q[1];
cx q[2],q[6];
cx q[10],q[3];
rz(pi/2) q[23];
cx q[24],q[2];
cx q[8],q[3];
cx q[6],q[4];
cx q[16],q[7];
cx q[10],q[9];
rx(pi/2) q[23];
cx q[4],q[1];
cx q[13],q[2];
cx q[17],q[9];
rz(pi/2) q[23];
cx q[8],q[2];
cx q[13],q[11];
cx q[6],q[11];
cx q[22],q[13];
cx q[4],q[6];
cx q[6],q[3];
cx q[4],q[3];
cx q[22],q[6];
cx q[1],q[4];
cx q[20],q[3];
cx q[21],q[6];
cx q[4],q[0];
cx q[18],q[3];
cx q[20],q[11];
cx q[2],q[0];
cx q[11],q[3];
cx q[21],q[4];
cx q[20],q[13];
cx q[24],q[0];
cx q[17],q[2];
cx q[5],q[3];
cx q[11],q[8];
cx q[18],q[13];
rz(pi/2) q[20];
cx q[21],q[23];
cx q[19],q[0];
cx q[6],q[2];
cx q[24],q[10];
rx(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[15],q[0];
cx q[12],q[10];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[24];
cx q[1],q[0];
cx q[12],q[7];
cx q[14],q[20];
rz(pi/2) q[15];
rx(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[24];
cx q[10],q[7];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[7];
rx(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[20];
cx q[22],q[24];
rx(pi/2) q[23];
cx q[1],q[15];
rx(pi/2) q[7];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[7];
rz(pi/2) q[15];
cx q[17],q[23];
rz(pi/2) q[20];
rx(pi/2) q[22];
rx(pi/2) q[24];
rx(pi/2) q[15];
rx(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[10],q[22];
rz(pi/2) q[15];
rz(pi/2) q[20];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[15];
rz(pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[15];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[15];
cx q[18],q[24];
rz(pi/2) q[22];
rx(pi/2) q[23];
cx q[18],q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[16],q[23];
cx q[18],q[19];
rz(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[24];
rz(pi/2) q[18];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[2],q[22];
rx(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[10],q[18];
rz(pi/2) q[19];
rx(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[23];
cx q[16],q[21];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[5],q[23];
cx q[16],q[19];
rz(pi/2) q[18];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[5];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[0],q[22];
rx(pi/2) q[5];
rx(pi/2) q[16];
rx(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[5];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[0],q[5];
cx q[13],q[16];
rz(pi/2) q[19];
rx(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[23];
rz(pi/2) q[0];
rz(pi/2) q[5];
cx q[13],q[14];
rz(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[0];
rx(pi/2) q[5];
cx q[11],q[21];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[16];
rz(pi/2) q[19];
cx q[22],q[24];
rz(pi/2) q[23];
rz(pi/2) q[0];
cx q[4],q[23];
rz(pi/2) q[5];
cx q[8],q[19];
rx(pi/2) q[13];
rx(pi/2) q[14];
cx q[24],q[15];
rz(pi/2) q[16];
rz(pi/2) q[21];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[5];
rz(pi/2) q[14];
rx(pi/2) q[15];
rx(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[5];
rx(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[23];
cx q[11],q[16];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[19];
rx(pi/2) q[21];
cx q[10],q[14];
cx q[11],q[13];
rx(pi/2) q[15];
rz(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[21];
cx q[10],q[4];
cx q[9],q[21];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[19];
cx q[3],q[19];
cx q[4],q[20];
rz(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[21];
rz(pi/2) q[3];
rx(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[3];
rz(pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[16];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[3];
cx q[24],q[9];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[20];
cx q[24],q[5];
cx q[8],q[16];
rz(pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[19];
cx q[1],q[14];
cx q[4],q[13];
rz(pi/2) q[5];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[16];
rx(pi/2) q[19];
cx q[22],q[24];
cx q[1],q[11];
rx(pi/2) q[5];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[1],q[3];
rz(pi/2) q[5];
rz(pi/2) q[8];
cx q[9],q[17];
rz(pi/2) q[11];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[16];
rx(pi/2) q[22];
rx(pi/2) q[24];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
cx q[17],q[15];
rz(pi/2) q[16];
rz(pi/2) q[22];
rz(pi/2) q[24];
rx(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[13];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[16];
cx q[21],q[3];
cx q[4],q[5];
cx q[6],q[16];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[15];
cx q[3],q[24];
cx q[10],q[4];
rz(pi/2) q[5];
cx q[6],q[7];
cx q[17],q[14];
rz(pi/2) q[16];
cx q[4],q[20];
rx(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[10];
cx q[17],q[12];
rz(pi/2) q[14];
rx(pi/2) q[16];
rz(pi/2) q[24];
rz(pi/2) q[5];
rx(pi/2) q[6];
rx(pi/2) q[7];
cx q[9],q[17];
rz(pi/2) q[10];
rz(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[16];
rx(pi/2) q[24];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[24];
rx(pi/2) q[5];
rz(pi/2) q[7];
rx(pi/2) q[9];
rz(pi/2) q[10];
rz(pi/2) q[12];
rz(pi/2) q[14];
rx(pi/2) q[16];
rz(pi/2) q[24];
rz(pi/2) q[5];
rx(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[12];
rx(pi/2) q[14];
rz(pi/2) q[16];
rx(pi/2) q[24];
rz(pi/2) q[7];
rx(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[24];
cx q[2],q[7];
cx q[3],q[14];
rz(pi/2) q[12];
cx q[20],q[24];
rz(pi/2) q[2];
cx q[3],q[5];
rz(pi/2) q[7];
rz(pi/2) q[14];
cx q[20],q[22];
rz(pi/2) q[24];
rx(pi/2) q[2];
cx q[21],q[3];
rz(pi/2) q[5];
rx(pi/2) q[7];
rx(pi/2) q[14];
cx q[20],q[16];
rz(pi/2) q[22];
rx(pi/2) q[24];
rz(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[24];
cx q[1],q[2];
rx(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[14];
rx(pi/2) q[16];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[1],q[23];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[7];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[24];
rx(pi/2) q[2];
rx(pi/2) q[5];
rz(pi/2) q[7];
rz(pi/2) q[14];
rz(pi/2) q[16];
cx q[23],q[18];
rx(pi/2) q[22];
rz(pi/2) q[24];
rz(pi/2) q[2];
rz(pi/2) q[5];
cx q[23],q[13];
rx(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[22];
cx q[17],q[2];
cx q[23],q[12];
rz(pi/2) q[13];
rz(pi/2) q[16];
rx(pi/2) q[18];
cx q[2],q[24];
cx q[23],q[6];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[18];
cx q[1],q[23];
rz(pi/2) q[6];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[18];
rz(pi/2) q[24];
rz(pi/2) q[1];
rx(pi/2) q[6];
rz(pi/2) q[12];
rz(pi/2) q[13];
rx(pi/2) q[18];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[1];
rz(pi/2) q[6];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[18];
rx(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[1];
cx q[2],q[18];
rz(pi/2) q[6];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[1];
cx q[2],q[9];
rx(pi/2) q[6];
rz(pi/2) q[12];
rz(pi/2) q[18];
rx(pi/2) q[24];
rz(pi/2) q[6];
rz(pi/2) q[9];
rx(pi/2) q[18];
rz(pi/2) q[24];
cx q[20],q[6];
rx(pi/2) q[9];
rz(pi/2) q[18];
cx q[20],q[3];
rz(pi/2) q[6];
rz(pi/2) q[9];
rz(pi/2) q[18];
rz(pi/2) q[3];
cx q[4],q[20];
rx(pi/2) q[6];
rz(pi/2) q[9];
rx(pi/2) q[18];
rx(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[6];
rx(pi/2) q[9];
rz(pi/2) q[18];
rz(pi/2) q[20];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[6];
rz(pi/2) q[9];
rx(pi/2) q[20];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[6];
rz(pi/2) q[20];
rx(pi/2) q[3];
rz(pi/2) q[6];
cx q[2],q[6];
rz(pi/2) q[3];
cx q[2],q[5];
rz(pi/2) q[6];
cx q[2],q[3];
rz(pi/2) q[5];
rx(pi/2) q[6];
cx q[2],q[0];
rz(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[0];
cx q[17],q[2];
rx(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[6];
rz(pi/2) q[17];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[17];
cx q[0],q[11];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[5];
rx(pi/2) q[17];
rz(pi/2) q[3];
cx q[11],q[20];
rz(pi/2) q[17];
cx q[11],q[19];
rz(pi/2) q[20];
cx q[11],q[14];
rz(pi/2) q[19];
rx(pi/2) q[20];
cx q[11],q[9];
rz(pi/2) q[14];
rx(pi/2) q[19];
rz(pi/2) q[20];
cx q[11],q[8];
rz(pi/2) q[9];
rx(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[20];
cx q[11],q[7];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[19];
rx(pi/2) q[20];
cx q[11],q[3];
rz(pi/2) q[7];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[19];
rz(pi/2) q[20];
cx q[0],q[11];
rz(pi/2) q[3];
rx(pi/2) q[7];
rz(pi/2) q[8];
rz(pi) q[9];
rx(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[3];
cx q[15],q[7];
rx(pi/2) q[9];
rz(pi/2) q[11];
rx(pi/2) q[0];
rz(pi/2) q[3];
cx q[7],q[23];
rz(pi/2) q[9];
rx(pi/2) q[11];
rz(pi/2) q[0];
rx(pi/2) q[3];
cx q[7],q[21];
rz(pi/2) q[11];
rz(pi/2) q[23];
rz(pi/2) q[3];
cx q[7],q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
cx q[7],q[16];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[23];
cx q[7],q[13];
rz(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[21];
rz(3*pi/2) q[23];
cx q[7],q[12];
rz(pi/2) q[13];
rx(pi/2) q[16];
rz(pi/2) q[19];
rz(pi) q[21];
rz(pi/2) q[23];
cx q[7],q[5];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(pi/2) q[16];
rz(pi) q[19];
rz(pi/2) q[21];
rx(pi/2) q[23];
cx q[7],q[3];
rz(pi/2) q[5];
rx(pi/2) q[12];
rz(pi/2) q[13];
rz(pi) q[16];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[3];
rx(pi/2) q[5];
cx q[15],q[7];
rz(pi/2) q[12];
rz(pi/2) q[13];
rz(pi/2) q[16];
rx(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[3];
rz(pi/2) q[5];
rz(pi) q[7];
cx q[12],q[8];
rx(pi/2) q[13];
rz(pi/2) q[15];
rx(pi/2) q[16];
rz(pi/2) q[19];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[7];
cx q[8],q[24];
rz(pi/2) q[13];
rz(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[3];
rx(pi/2) q[5];
rx(pi/2) q[7];
cx q[8],q[22];
rx(pi/2) q[15];
rz(pi/2) q[24];
rx(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[7];
cx q[8],q[20];
rz(pi/2) q[15];
rz(pi/2) q[22];
rx(pi/2) q[24];
rz(pi/2) q[3];
cx q[8],q[18];
rz(pi/2) q[20];
rx(pi/2) q[22];
rz(pi/2) q[24];
cx q[8],q[13];
rz(pi/2) q[18];
rx(pi/2) q[20];
rz(pi/2) q[22];
rz(pi/2) q[24];
cx q[8],q[6];
rz(pi/2) q[13];
rx(pi/2) q[18];
rz(pi/2) q[20];
rz(pi/2) q[22];
rx(pi/2) q[24];
cx q[8],q[5];
rz(pi/2) q[6];
rx(pi/2) q[13];
rz(pi/2) q[18];
rz(pi) q[20];
rx(pi/2) q[22];
rz(pi/2) q[24];
cx q[8],q[4];
rz(pi/2) q[5];
rx(pi/2) q[6];
rz(pi/2) q[13];
rz(3*pi/2) q[18];
rz(pi/2) q[22];
cx q[8],q[3];
rz(pi/2) q[4];
rx(pi/2) q[5];
rz(pi/2) q[6];
rz(pi) q[13];
rz(pi/2) q[18];
cx q[8],q[2];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[5];
rz(3*pi/2) q[6];
rx(pi/2) q[18];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[4];
rz(pi) q[5];
rz(pi/2) q[6];
cx q[12],q[8];
rz(pi/2) q[18];
rx(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[6];
rz(pi/2) q[12];
rz(pi/2) q[2];
rz(pi) q[3];
rx(pi/2) q[4];
rx(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[12];
rz(3*pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[12];
rz(pi/2) q[2];
rz(pi/2) q[12];
rx(pi/2) q[2];
rz(pi/2) q[2];
