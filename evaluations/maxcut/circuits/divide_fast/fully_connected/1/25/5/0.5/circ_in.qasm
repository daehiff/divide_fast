OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
rx(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[6];
rx(pi/2) q[7];
rx(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[10];
rx(pi/2) q[11];
rx(pi/2) q[12];
rx(pi/2) q[13];
rx(pi/2) q[14];
rx(pi/2) q[15];
rx(pi/2) q[16];
rx(pi/2) q[17];
rx(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[23];
rx(pi/2) q[24];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[2],q[0];
rz(pi/2) q[0];
cx q[2],q[0];
cx q[4],q[0];
rz(pi/2) q[0];
cx q[4],q[0];
cx q[6],q[0];
rz(pi/2) q[0];
cx q[6],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[10],q[0];
rz(pi/2) q[0];
cx q[10],q[0];
cx q[11],q[0];
rz(pi/2) q[0];
cx q[11],q[0];
cx q[12],q[0];
rz(pi/2) q[0];
cx q[12],q[0];
cx q[15],q[0];
rz(pi/2) q[0];
cx q[15],q[0];
cx q[19],q[0];
rz(pi/2) q[0];
cx q[19],q[0];
cx q[21],q[0];
rz(pi/2) q[0];
cx q[21],q[0];
cx q[24],q[0];
rz(pi/2) q[0];
cx q[24],q[0];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[5],q[1];
rz(pi/2) q[1];
cx q[5],q[1];
cx q[9],q[1];
rz(pi/2) q[1];
cx q[9],q[1];
cx q[11],q[1];
rz(pi/2) q[1];
cx q[11],q[1];
cx q[13],q[1];
rz(pi/2) q[1];
cx q[13],q[1];
cx q[14],q[1];
rz(pi/2) q[1];
cx q[14],q[1];
cx q[16],q[1];
rz(pi/2) q[1];
cx q[16],q[1];
cx q[17],q[1];
rz(pi/2) q[1];
cx q[17],q[1];
cx q[19],q[1];
rz(pi/2) q[1];
cx q[19],q[1];
cx q[21],q[1];
rz(pi/2) q[1];
cx q[21],q[1];
cx q[22],q[1];
rz(pi/2) q[1];
cx q[22],q[1];
cx q[23],q[1];
rz(pi/2) q[1];
cx q[23],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[6],q[2];
rz(pi/2) q[2];
cx q[6],q[2];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[10],q[2];
rz(pi/2) q[2];
cx q[10],q[2];
cx q[19],q[2];
rz(pi/2) q[2];
cx q[19],q[2];
cx q[20],q[2];
rz(pi/2) q[2];
cx q[20],q[2];
cx q[21],q[2];
rz(pi/2) q[2];
cx q[21],q[2];
cx q[23],q[2];
rz(pi/2) q[2];
cx q[23],q[2];
cx q[5],q[3];
rz(pi/2) q[3];
cx q[5],q[3];
cx q[9],q[3];
rz(pi/2) q[3];
cx q[9],q[3];
cx q[13],q[3];
rz(pi/2) q[3];
cx q[13],q[3];
cx q[14],q[3];
rz(pi/2) q[3];
cx q[14],q[3];
cx q[18],q[3];
rz(pi/2) q[3];
cx q[18],q[3];
cx q[19],q[3];
rz(pi/2) q[3];
cx q[19],q[3];
cx q[20],q[3];
rz(pi/2) q[3];
cx q[20],q[3];
cx q[22],q[3];
rz(pi/2) q[3];
cx q[22],q[3];
cx q[23],q[3];
rz(pi/2) q[3];
cx q[23],q[3];
cx q[24],q[3];
rz(pi/2) q[3];
cx q[24],q[3];
cx q[7],q[4];
rz(pi/2) q[4];
cx q[7],q[4];
cx q[8],q[4];
rz(pi/2) q[4];
cx q[8],q[4];
cx q[9],q[4];
rz(pi/2) q[4];
cx q[9],q[4];
cx q[11],q[4];
rz(pi/2) q[4];
cx q[11],q[4];
cx q[12],q[4];
rz(pi/2) q[4];
cx q[12],q[4];
cx q[14],q[4];
rz(pi/2) q[4];
cx q[14],q[4];
cx q[15],q[4];
rz(pi/2) q[4];
cx q[15],q[4];
cx q[17],q[4];
rz(pi/2) q[4];
cx q[17],q[4];
cx q[19],q[4];
rz(pi/2) q[4];
cx q[19],q[4];
cx q[23],q[4];
rz(pi/2) q[4];
cx q[23],q[4];
cx q[24],q[4];
rz(pi/2) q[4];
cx q[24],q[4];
cx q[7],q[5];
rz(pi/2) q[5];
cx q[7],q[5];
cx q[13],q[5];
rz(pi/2) q[5];
cx q[13],q[5];
cx q[16],q[5];
rz(pi/2) q[5];
cx q[16],q[5];
cx q[17],q[5];
rz(pi/2) q[5];
cx q[17],q[5];
cx q[18],q[5];
rz(pi/2) q[5];
cx q[18],q[5];
cx q[21],q[5];
rz(pi/2) q[5];
cx q[21],q[5];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[8],q[6];
rz(pi/2) q[6];
cx q[8],q[6];
cx q[11],q[6];
rz(pi/2) q[6];
cx q[11],q[6];
cx q[12],q[6];
rz(pi/2) q[6];
cx q[12],q[6];
cx q[15],q[6];
rz(pi/2) q[6];
cx q[15],q[6];
cx q[18],q[6];
rz(pi/2) q[6];
cx q[18],q[6];
cx q[19],q[6];
rz(pi/2) q[6];
cx q[19],q[6];
cx q[8],q[7];
rz(pi/2) q[7];
cx q[8],q[7];
cx q[10],q[7];
rz(pi/2) q[7];
cx q[10],q[7];
cx q[12],q[7];
rz(pi/2) q[7];
cx q[12],q[7];
cx q[16],q[7];
rz(pi/2) q[7];
cx q[16],q[7];
cx q[17],q[7];
rz(pi/2) q[7];
cx q[17],q[7];
cx q[19],q[7];
rz(pi/2) q[7];
cx q[19],q[7];
cx q[21],q[7];
rz(pi/2) q[7];
cx q[21],q[7];
cx q[22],q[7];
rz(pi/2) q[7];
cx q[22],q[7];
cx q[23],q[7];
rz(pi/2) q[7];
cx q[23],q[7];
cx q[24],q[7];
rz(pi/2) q[7];
cx q[24],q[7];
cx q[9],q[8];
rz(pi/2) q[8];
cx q[9],q[8];
cx q[10],q[8];
rz(pi/2) q[8];
cx q[10],q[8];
cx q[13],q[8];
rz(pi/2) q[8];
cx q[13],q[8];
cx q[16],q[8];
rz(pi/2) q[8];
cx q[16],q[8];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[21],q[8];
rz(pi/2) q[8];
cx q[21],q[8];
cx q[22],q[8];
rz(pi/2) q[8];
cx q[22],q[8];
cx q[12],q[9];
rz(pi/2) q[9];
cx q[12],q[9];
cx q[13],q[9];
rz(pi/2) q[9];
cx q[13],q[9];
cx q[14],q[9];
rz(pi/2) q[9];
cx q[14],q[9];
cx q[16],q[9];
rz(pi/2) q[9];
cx q[16],q[9];
cx q[20],q[9];
rz(pi/2) q[9];
cx q[20],q[9];
cx q[24],q[9];
rz(pi/2) q[9];
cx q[24],q[9];
cx q[11],q[10];
rz(pi/2) q[10];
cx q[11],q[10];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[15],q[10];
rz(pi/2) q[10];
cx q[15],q[10];
cx q[22],q[10];
rz(pi/2) q[10];
cx q[22],q[10];
cx q[23],q[10];
rz(pi/2) q[10];
cx q[23],q[10];
cx q[24],q[10];
rz(pi/2) q[10];
cx q[24],q[10];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[17],q[11];
rz(pi/2) q[11];
cx q[17],q[11];
cx q[18],q[11];
rz(pi/2) q[11];
cx q[18],q[11];
cx q[19],q[11];
rz(pi/2) q[11];
cx q[19],q[11];
cx q[20],q[11];
rz(pi/2) q[11];
cx q[20],q[11];
cx q[21],q[11];
rz(pi/2) q[11];
cx q[21],q[11];
cx q[24],q[11];
rz(pi/2) q[11];
cx q[24],q[11];
cx q[16],q[12];
rz(pi/2) q[12];
cx q[16],q[12];
cx q[19],q[12];
rz(pi/2) q[12];
cx q[19],q[12];
cx q[20],q[12];
rz(pi/2) q[12];
cx q[20],q[12];
cx q[22],q[12];
rz(pi/2) q[12];
cx q[22],q[12];
cx q[24],q[12];
rz(pi/2) q[12];
cx q[24],q[12];
cx q[14],q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[15],q[13];
rz(pi/2) q[13];
cx q[15],q[13];
cx q[16],q[13];
rz(pi/2) q[13];
cx q[16],q[13];
cx q[19],q[13];
rz(pi/2) q[13];
cx q[19],q[13];
cx q[20],q[13];
rz(pi/2) q[13];
cx q[20],q[13];
cx q[23],q[13];
rz(pi/2) q[13];
cx q[23],q[13];
cx q[15],q[14];
rz(pi/2) q[14];
cx q[15],q[14];
cx q[16],q[14];
rz(pi/2) q[14];
cx q[16],q[14];
cx q[18],q[14];
rz(pi/2) q[14];
cx q[18],q[14];
cx q[19],q[14];
rz(pi/2) q[14];
cx q[19],q[14];
cx q[21],q[14];
rz(pi/2) q[14];
cx q[21],q[14];
cx q[24],q[14];
rz(pi/2) q[14];
cx q[24],q[14];
cx q[16],q[15];
rz(pi/2) q[15];
cx q[16],q[15];
cx q[21],q[15];
rz(pi/2) q[15];
cx q[21],q[15];
cx q[22],q[15];
rz(pi/2) q[15];
cx q[22],q[15];
cx q[17],q[16];
rz(pi/2) q[16];
cx q[17],q[16];
cx q[21],q[16];
rz(pi/2) q[16];
cx q[21],q[16];
cx q[23],q[17];
rz(pi/2) q[17];
cx q[23],q[17];
cx q[24],q[17];
rz(pi/2) q[17];
cx q[24],q[17];
cx q[19],q[18];
rz(pi/2) q[18];
cx q[19],q[18];
cx q[20],q[18];
rz(pi/2) q[18];
cx q[20],q[18];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
cx q[22],q[18];
rz(pi/2) q[18];
cx q[22],q[18];
cx q[23],q[18];
rz(pi/2) q[18];
cx q[23],q[18];
cx q[24],q[18];
rz(pi/2) q[18];
cx q[24],q[18];
cx q[22],q[19];
rz(pi/2) q[19];
cx q[22],q[19];
cx q[24],q[19];
rz(pi/2) q[19];
cx q[24],q[19];
cx q[22],q[20];
rz(pi/2) q[20];
cx q[22],q[20];
cx q[23],q[20];
rz(pi/2) q[20];
cx q[23],q[20];
cx q[24],q[20];
rz(pi/2) q[20];
cx q[24],q[20];
cx q[22],q[21];
rz(pi/2) q[21];
cx q[22],q[21];
cx q[23],q[21];
rz(pi/2) q[21];
cx q[23],q[21];
cx q[24],q[21];
rz(pi/2) q[21];
cx q[24],q[21];
cx q[24],q[23];
rz(pi/2) q[23];
cx q[24],q[23];
