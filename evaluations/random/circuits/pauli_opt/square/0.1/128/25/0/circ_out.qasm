OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[9],q[10];
cx q[12],q[17];
cx q[1],q[2];
cx q[16],q[13];
cx q[24],q[15];
cx q[18],q[11];
cx q[15],q[24];
cx q[2],q[1];
cx q[6],q[13];
cx q[21],q[18];
cx q[10],q[9];
cx q[21],q[22];
cx q[24],q[15];
cx q[0],q[9];
cx q[3],q[4];
rz(pi) q[0];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[8];
rz(pi) q[24];
rz(pi) q[19];
rz(pi) q[23];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[1];
rx(pi) q[8];
rx(pi) q[10];
rx(pi) q[16];
rx(pi) q[20];
cx q[18],q[3];
rz(7*pi/4) q[3];
cx q[18],q[3];
rz(7*pi/4) q[19];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[24],q[18];
cx q[18],q[11];
rz(5*pi/4) q[11];
cx q[18],q[11];
cx q[24],q[18];
cx q[22],q[20];
rz(7*pi/4) q[20];
cx q[22],q[20];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(3*pi/4) q[0];
cx q[22],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/2) q[20];
rz(7*pi/4) q[22];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[15];
rx(pi/2) q[5];
cx q[5],q[15];
cx q[15],q[16];
rx(pi/4) q[15];
cx q[15],q[16];
rx(pi/4) q[14];
rx(5*pi/4) q[17];
cx q[13],q[14];
cx q[6],q[13];
rx(3*pi/2) q[6];
cx q[6],q[13];
cx q[13],q[14];
cx q[15],q[17];
rx(7*pi/4) q[15];
cx q[15],q[17];
rx(pi/2) q[11];
cx q[11],q[18];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[12],q[14];
rx(3*pi/4) q[12];
cx q[12],q[14];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
rx(5*pi/4) q[19];
rx(pi/2) q[24];
cx q[23],q[24];
rx(3*pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[5];
rz(pi/2) q[15];
cx q[5],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[5],q[7];
rz(3*pi/4) q[12];
rz(7*pi/4) q[11];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
rz(5*pi/4) q[6];
cx q[21],q[12];
rz(pi/2) q[12];
cx q[21],q[12];
cx q[22],q[14];
rz(7*pi/4) q[14];
cx q[22],q[14];
cx q[4],q[3];
cx q[18],q[3];
rz(5*pi/4) q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[8],q[7];
rz(5*pi/4) q[7];
cx q[8],q[7];
cx q[18],q[11];
cx q[11],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[11],q[3];
cx q[18],q[11];
rz(pi/4) q[7];
cx q[12],q[5];
rz(pi/4) q[5];
cx q[12],q[5];
cx q[9],q[0];
cx q[3],q[0];
rz(3*pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
rz(5*pi/4) q[23];
cx q[19],q[3];
rz(pi/4) q[3];
cx q[19],q[3];
rx(pi/4) q[14];
cx q[15],q[24];
rx(pi/4) q[15];
cx q[15],q[24];
cx q[10],q[15];
rx(3*pi/2) q[10];
cx q[10],q[15];
cx q[6],q[13];
rx(pi/4) q[6];
cx q[6],q[13];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[21],q[24];
rx(pi/4) q[21];
cx q[21],q[24];
cx q[11],q[18];
cx q[8],q[11];
rx(3*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
rx(3*pi/4) q[11];
rx(7*pi/4) q[19];
cx q[7],q[20];
cx q[2],q[7];
rx(5*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[20];
rx(3*pi/4) q[23];
cx q[10],q[9];
cx q[9],q[0];
rz(pi/4) q[0];
cx q[9],q[0];
cx q[10],q[9];
rz(pi/2) q[15];
rz(5*pi/4) q[6];
cx q[10],q[9];
cx q[4],q[3];
cx q[9],q[0];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[4],q[3];
cx q[10],q[9];
cx q[22],q[7];
cx q[7],q[2];
rz(3*pi/4) q[2];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/4) q[8];
rz(pi/4) q[21];
cx q[13],q[6];
cx q[6],q[7];
cx q[7],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[13],q[6];
rx(3*pi/2) q[10];
cx q[6],q[13];
cx q[9],q[6];
cx q[0],q[9];
rx(7*pi/4) q[0];
cx q[0],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[18];
cx q[13],q[11];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[11];
cx q[11],q[18];
cx q[2],q[13];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[13];
cx q[10],q[21];
cx q[9],q[10];
rx(pi/4) q[9];
cx q[9],q[10];
cx q[10],q[21];
cx q[9],q[10];
cx q[8],q[9];
rx(5*pi/4) q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[16];
rx(5*pi/4) q[10];
cx q[10],q[16];
rx(3*pi/4) q[8];
rx(7*pi/4) q[15];
rx(5*pi/4) q[13];
rx(pi/2) q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(5*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[9],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[15],q[18];
cx q[18],q[11];
rz(pi/4) q[11];
cx q[18],q[11];
cx q[15],q[18];
rz(3*pi/4) q[18];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[2],q[12];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
cx q[2],q[12];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
cx q[1],q[11];
rx(3*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[8];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[8];
rx(pi/4) q[15];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[11],q[14];
rx(3*pi/4) q[11];
cx q[11],q[14];
rz(pi/2) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[18],q[15];
cx q[7],q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[6],q[13];
rx(7*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[12];
rz(pi) q[0];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[8];
rz(pi) q[24];
rz(pi) q[19];
rz(pi) q[23];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[1];
rx(pi) q[8];
rx(pi) q[10];
rx(pi) q[16];
rx(pi) q[20];
cx q[18],q[3];
rz(7*pi/4) q[3];
cx q[18],q[3];
rz(7*pi/4) q[19];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[24],q[18];
cx q[18],q[11];
rz(5*pi/4) q[11];
cx q[18],q[11];
cx q[24],q[18];
cx q[22],q[20];
rz(7*pi/4) q[20];
cx q[22],q[20];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(3*pi/4) q[0];
cx q[22],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/2) q[20];
rz(7*pi/4) q[22];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[15];
rx(pi/2) q[5];
cx q[5],q[15];
cx q[15],q[16];
rx(pi/4) q[15];
cx q[15],q[16];
rx(pi/4) q[14];
rx(5*pi/4) q[17];
cx q[13],q[14];
cx q[6],q[13];
rx(3*pi/2) q[6];
cx q[6],q[13];
cx q[13],q[14];
cx q[15],q[17];
rx(7*pi/4) q[15];
cx q[15],q[17];
rx(pi/2) q[11];
cx q[11],q[18];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[12],q[14];
rx(3*pi/4) q[12];
cx q[12],q[14];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
rx(5*pi/4) q[19];
rx(pi/2) q[24];
cx q[23],q[24];
rx(3*pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[5];
rz(pi/2) q[15];
cx q[5],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[5],q[7];
rz(3*pi/4) q[12];
rz(7*pi/4) q[11];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
rz(5*pi/4) q[6];
cx q[21],q[12];
rz(pi/2) q[12];
cx q[21],q[12];
cx q[22],q[14];
rz(7*pi/4) q[14];
cx q[22],q[14];
cx q[4],q[3];
cx q[18],q[3];
rz(5*pi/4) q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[8],q[7];
rz(5*pi/4) q[7];
cx q[8],q[7];
cx q[18],q[11];
cx q[11],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[11],q[3];
cx q[18],q[11];
rz(pi/4) q[7];
cx q[12],q[5];
rz(pi/4) q[5];
cx q[12],q[5];
cx q[9],q[0];
cx q[3],q[0];
rz(3*pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
rz(5*pi/4) q[23];
cx q[19],q[3];
rz(pi/4) q[3];
cx q[19],q[3];
rx(pi/4) q[14];
cx q[15],q[24];
rx(pi/4) q[15];
cx q[15],q[24];
cx q[10],q[15];
rx(3*pi/2) q[10];
cx q[10],q[15];
cx q[6],q[13];
rx(pi/4) q[6];
cx q[6],q[13];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[21],q[24];
rx(pi/4) q[21];
cx q[21],q[24];
cx q[11],q[18];
cx q[8],q[11];
rx(3*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
rx(3*pi/4) q[11];
rx(7*pi/4) q[19];
cx q[7],q[20];
cx q[2],q[7];
rx(5*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[20];
rx(3*pi/4) q[23];
cx q[10],q[9];
cx q[9],q[0];
rz(pi/4) q[0];
cx q[9],q[0];
cx q[10],q[9];
rz(pi/2) q[15];
rz(5*pi/4) q[6];
cx q[10],q[9];
cx q[4],q[3];
cx q[9],q[0];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[4],q[3];
cx q[10],q[9];
cx q[22],q[7];
cx q[7],q[2];
rz(3*pi/4) q[2];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/4) q[8];
rz(pi/4) q[21];
cx q[13],q[6];
cx q[6],q[7];
cx q[7],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[13],q[6];
rx(3*pi/2) q[10];
cx q[6],q[13];
cx q[9],q[6];
cx q[0],q[9];
rx(7*pi/4) q[0];
cx q[0],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[18];
cx q[13],q[11];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[11];
cx q[11],q[18];
cx q[2],q[13];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[13];
cx q[10],q[21];
cx q[9],q[10];
rx(pi/4) q[9];
cx q[9],q[10];
cx q[10],q[21];
cx q[9],q[10];
cx q[8],q[9];
rx(5*pi/4) q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[16];
rx(5*pi/4) q[10];
cx q[10],q[16];
rx(3*pi/4) q[8];
rx(7*pi/4) q[15];
rx(5*pi/4) q[13];
rx(pi/2) q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(5*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[9],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[15],q[18];
cx q[18],q[11];
rz(pi/4) q[11];
cx q[18],q[11];
cx q[15],q[18];
rz(3*pi/4) q[18];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[2],q[12];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
cx q[2],q[12];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
cx q[1],q[11];
rx(3*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[8];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[8];
rx(pi/4) q[15];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[11],q[14];
rx(3*pi/4) q[11];
cx q[11],q[14];
rz(pi/2) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[18],q[15];
cx q[7],q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[6],q[13];
rx(7*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[12];
rz(pi) q[0];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[8];
rz(pi) q[24];
rz(pi) q[19];
rz(pi) q[23];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[1];
rx(pi) q[8];
rx(pi) q[10];
rx(pi) q[16];
rx(pi) q[20];
cx q[18],q[3];
rz(7*pi/4) q[3];
cx q[18],q[3];
rz(7*pi/4) q[19];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[24],q[18];
cx q[18],q[11];
rz(5*pi/4) q[11];
cx q[18],q[11];
cx q[24],q[18];
cx q[22],q[20];
rz(7*pi/4) q[20];
cx q[22],q[20];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(3*pi/4) q[0];
cx q[22],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/2) q[20];
rz(7*pi/4) q[22];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[15];
rx(pi/2) q[5];
cx q[5],q[15];
cx q[15],q[16];
rx(pi/4) q[15];
cx q[15],q[16];
rx(pi/4) q[14];
rx(5*pi/4) q[17];
cx q[13],q[14];
cx q[6],q[13];
rx(3*pi/2) q[6];
cx q[6],q[13];
cx q[13],q[14];
cx q[15],q[17];
rx(7*pi/4) q[15];
cx q[15],q[17];
rx(pi/2) q[11];
cx q[11],q[18];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[12],q[14];
rx(3*pi/4) q[12];
cx q[12],q[14];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
rx(5*pi/4) q[19];
rx(pi/2) q[24];
cx q[23],q[24];
rx(3*pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[5];
rz(pi/2) q[15];
cx q[5],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[5],q[7];
rz(3*pi/4) q[12];
rz(7*pi/4) q[11];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
rz(5*pi/4) q[6];
cx q[21],q[12];
rz(pi/2) q[12];
cx q[21],q[12];
cx q[22],q[14];
rz(7*pi/4) q[14];
cx q[22],q[14];
cx q[4],q[3];
cx q[18],q[3];
rz(5*pi/4) q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[8],q[7];
rz(5*pi/4) q[7];
cx q[8],q[7];
cx q[18],q[11];
cx q[11],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[11],q[3];
cx q[18],q[11];
rz(pi/4) q[7];
cx q[12],q[5];
rz(pi/4) q[5];
cx q[12],q[5];
cx q[9],q[0];
cx q[3],q[0];
rz(3*pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
rz(5*pi/4) q[23];
cx q[19],q[3];
rz(pi/4) q[3];
cx q[19],q[3];
rx(pi/4) q[14];
cx q[15],q[24];
rx(pi/4) q[15];
cx q[15],q[24];
cx q[10],q[15];
rx(3*pi/2) q[10];
cx q[10],q[15];
cx q[6],q[13];
rx(pi/4) q[6];
cx q[6],q[13];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[21],q[24];
rx(pi/4) q[21];
cx q[21],q[24];
cx q[11],q[18];
cx q[8],q[11];
rx(3*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
rx(3*pi/4) q[11];
rx(7*pi/4) q[19];
cx q[7],q[20];
cx q[2],q[7];
rx(5*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[20];
rx(3*pi/4) q[23];
cx q[10],q[9];
cx q[9],q[0];
rz(pi/4) q[0];
cx q[9],q[0];
cx q[10],q[9];
rz(pi/2) q[15];
rz(5*pi/4) q[6];
cx q[10],q[9];
cx q[4],q[3];
cx q[9],q[0];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[4],q[3];
cx q[10],q[9];
cx q[22],q[7];
cx q[7],q[2];
rz(3*pi/4) q[2];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/4) q[8];
rz(pi/4) q[21];
cx q[13],q[6];
cx q[6],q[7];
cx q[7],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[13],q[6];
rx(3*pi/2) q[10];
cx q[6],q[13];
cx q[9],q[6];
cx q[0],q[9];
rx(7*pi/4) q[0];
cx q[0],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[18];
cx q[13],q[11];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[11];
cx q[11],q[18];
cx q[2],q[13];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[13];
cx q[10],q[21];
cx q[9],q[10];
rx(pi/4) q[9];
cx q[9],q[10];
cx q[10],q[21];
cx q[9],q[10];
cx q[8],q[9];
rx(5*pi/4) q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[16];
rx(5*pi/4) q[10];
cx q[10],q[16];
rx(3*pi/4) q[8];
rx(7*pi/4) q[15];
rx(5*pi/4) q[13];
rx(pi/2) q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(5*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[9],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[15],q[18];
cx q[18],q[11];
rz(pi/4) q[11];
cx q[18],q[11];
cx q[15],q[18];
rz(3*pi/4) q[18];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[2],q[12];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
cx q[2],q[12];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
cx q[1],q[11];
rx(3*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[8];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[8];
rx(pi/4) q[15];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[11],q[14];
rx(3*pi/4) q[11];
cx q[11],q[14];
rz(pi/2) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[18],q[15];
cx q[7],q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[6],q[13];
rx(7*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[12];
rz(pi) q[0];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[8];
rz(pi) q[24];
rz(pi) q[19];
rz(pi) q[23];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[1];
rx(pi) q[8];
rx(pi) q[10];
rx(pi) q[16];
rx(pi) q[20];
cx q[18],q[3];
rz(7*pi/4) q[3];
cx q[18],q[3];
rz(7*pi/4) q[19];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[24],q[18];
cx q[18],q[11];
rz(5*pi/4) q[11];
cx q[18],q[11];
cx q[24],q[18];
cx q[22],q[20];
rz(7*pi/4) q[20];
cx q[22],q[20];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(3*pi/4) q[0];
cx q[22],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/2) q[20];
rz(7*pi/4) q[22];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[15];
rx(pi/2) q[5];
cx q[5],q[15];
cx q[15],q[16];
rx(pi/4) q[15];
cx q[15],q[16];
rx(pi/4) q[14];
rx(5*pi/4) q[17];
cx q[13],q[14];
cx q[6],q[13];
rx(3*pi/2) q[6];
cx q[6],q[13];
cx q[13],q[14];
cx q[15],q[17];
rx(7*pi/4) q[15];
cx q[15],q[17];
rx(pi/2) q[11];
cx q[11],q[18];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[12],q[14];
rx(3*pi/4) q[12];
cx q[12],q[14];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
rx(5*pi/4) q[19];
rx(pi/2) q[24];
cx q[23],q[24];
rx(3*pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[5];
rz(pi/2) q[15];
cx q[5],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[5],q[7];
rz(3*pi/4) q[12];
rz(7*pi/4) q[11];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
rz(5*pi/4) q[6];
cx q[21],q[12];
rz(pi/2) q[12];
cx q[21],q[12];
cx q[22],q[14];
rz(7*pi/4) q[14];
cx q[22],q[14];
cx q[4],q[3];
cx q[18],q[3];
rz(5*pi/4) q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[8],q[7];
rz(5*pi/4) q[7];
cx q[8],q[7];
cx q[18],q[11];
cx q[11],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[11],q[3];
cx q[18],q[11];
rz(pi/4) q[7];
cx q[12],q[5];
rz(pi/4) q[5];
cx q[12],q[5];
cx q[9],q[0];
cx q[3],q[0];
rz(3*pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
rz(5*pi/4) q[23];
cx q[19],q[3];
rz(pi/4) q[3];
cx q[19],q[3];
rx(pi/4) q[14];
cx q[15],q[24];
rx(pi/4) q[15];
cx q[15],q[24];
cx q[10],q[15];
rx(3*pi/2) q[10];
cx q[10],q[15];
cx q[6],q[13];
rx(pi/4) q[6];
cx q[6],q[13];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[21],q[24];
rx(pi/4) q[21];
cx q[21],q[24];
cx q[11],q[18];
cx q[8],q[11];
rx(3*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
rx(3*pi/4) q[11];
rx(7*pi/4) q[19];
cx q[7],q[20];
cx q[2],q[7];
rx(5*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[20];
rx(3*pi/4) q[23];
cx q[10],q[9];
cx q[9],q[0];
rz(pi/4) q[0];
cx q[9],q[0];
cx q[10],q[9];
rz(pi/2) q[15];
rz(5*pi/4) q[6];
cx q[10],q[9];
cx q[4],q[3];
cx q[9],q[0];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[4],q[3];
cx q[10],q[9];
cx q[22],q[7];
cx q[7],q[2];
rz(3*pi/4) q[2];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/4) q[8];
rz(pi/4) q[21];
cx q[13],q[6];
cx q[6],q[7];
cx q[7],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[13],q[6];
rx(3*pi/2) q[10];
cx q[6],q[13];
cx q[9],q[6];
cx q[0],q[9];
rx(7*pi/4) q[0];
cx q[0],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[18];
cx q[13],q[11];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[11];
cx q[11],q[18];
cx q[2],q[13];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[13];
cx q[10],q[21];
cx q[9],q[10];
rx(pi/4) q[9];
cx q[9],q[10];
cx q[10],q[21];
cx q[9],q[10];
cx q[8],q[9];
rx(5*pi/4) q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[16];
rx(5*pi/4) q[10];
cx q[10],q[16];
rx(3*pi/4) q[8];
rx(7*pi/4) q[15];
rx(5*pi/4) q[13];
rx(pi/2) q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(5*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[9],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[15],q[18];
cx q[18],q[11];
rz(pi/4) q[11];
cx q[18],q[11];
cx q[15],q[18];
rz(3*pi/4) q[18];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[2],q[12];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
cx q[2],q[12];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
cx q[1],q[11];
rx(3*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[8];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[8];
rx(pi/4) q[15];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[11],q[14];
rx(3*pi/4) q[11];
cx q[11],q[14];
rz(pi/2) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[18],q[15];
cx q[7],q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[6],q[13];
rx(7*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[12];
rz(pi) q[0];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[3];
cx q[4],q[3];
rz(pi) q[8];
rz(pi) q[24];
rz(pi) q[19];
rz(pi) q[23];
cx q[1],q[2];
rx(pi) q[1];
cx q[1],q[2];
rx(pi) q[1];
rx(pi) q[8];
rx(pi) q[10];
rx(pi) q[16];
rx(pi) q[20];
cx q[18],q[3];
rz(7*pi/4) q[3];
cx q[18],q[3];
rz(7*pi/4) q[19];
cx q[19],q[8];
rz(pi/2) q[8];
cx q[19],q[8];
cx q[24],q[18];
cx q[18],q[11];
rz(5*pi/4) q[11];
cx q[18],q[11];
cx q[24],q[18];
cx q[22],q[20];
rz(7*pi/4) q[20];
cx q[22],q[20];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(3*pi/4) q[0];
cx q[22],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/2) q[20];
rz(7*pi/4) q[22];
cx q[4],q[3];
rz(3*pi/2) q[3];
cx q[4],q[3];
cx q[5],q[15];
rx(pi/2) q[5];
cx q[5],q[15];
cx q[15],q[16];
rx(pi/4) q[15];
cx q[15],q[16];
rx(pi/4) q[14];
rx(5*pi/4) q[17];
cx q[13],q[14];
cx q[6],q[13];
rx(3*pi/2) q[6];
cx q[6],q[13];
cx q[13],q[14];
cx q[15],q[17];
rx(7*pi/4) q[15];
cx q[15],q[17];
rx(pi/2) q[11];
cx q[11],q[18];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[12],q[14];
rx(3*pi/4) q[12];
cx q[12],q[14];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
rx(5*pi/4) q[19];
rx(pi/2) q[24];
cx q[23],q[24];
rx(3*pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[5];
rz(pi/2) q[15];
cx q[5],q[7];
cx q[7],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[7],q[2];
cx q[5],q[7];
rz(3*pi/4) q[12];
rz(7*pi/4) q[11];
cx q[21],q[18];
rz(pi/2) q[18];
cx q[21],q[18];
rz(5*pi/4) q[6];
cx q[21],q[12];
rz(pi/2) q[12];
cx q[21],q[12];
cx q[22],q[14];
rz(7*pi/4) q[14];
cx q[22],q[14];
cx q[4],q[3];
cx q[18],q[3];
rz(5*pi/4) q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[8],q[7];
rz(5*pi/4) q[7];
cx q[8],q[7];
cx q[18],q[11];
cx q[11],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[11],q[3];
cx q[18],q[11];
rz(pi/4) q[7];
cx q[12],q[5];
rz(pi/4) q[5];
cx q[12],q[5];
cx q[9],q[0];
cx q[3],q[0];
rz(3*pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
rz(5*pi/4) q[23];
cx q[19],q[3];
rz(pi/4) q[3];
cx q[19],q[3];
rx(pi/4) q[14];
cx q[15],q[24];
rx(pi/4) q[15];
cx q[15],q[24];
cx q[10],q[15];
rx(3*pi/2) q[10];
cx q[10],q[15];
cx q[6],q[13];
rx(pi/4) q[6];
cx q[6],q[13];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[21],q[24];
rx(pi/4) q[21];
cx q[21],q[24];
cx q[11],q[18];
cx q[8],q[11];
rx(3*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[18];
rx(3*pi/4) q[11];
rx(7*pi/4) q[19];
cx q[7],q[20];
cx q[2],q[7];
rx(5*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[20];
rx(3*pi/4) q[23];
cx q[10],q[9];
cx q[9],q[0];
rz(pi/4) q[0];
cx q[9],q[0];
cx q[10],q[9];
rz(pi/2) q[15];
rz(5*pi/4) q[6];
cx q[10],q[9];
cx q[4],q[3];
cx q[9],q[0];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[4],q[3];
cx q[10],q[9];
cx q[22],q[7];
cx q[7],q[2];
rz(3*pi/4) q[2];
cx q[7],q[2];
cx q[22],q[7];
rz(3*pi/4) q[8];
rz(pi/4) q[21];
cx q[13],q[6];
cx q[6],q[7];
cx q[7],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[13],q[6];
rx(3*pi/2) q[10];
cx q[6],q[13];
cx q[9],q[6];
cx q[0],q[9];
rx(7*pi/4) q[0];
cx q[0],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[18];
cx q[13],q[11];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[11];
cx q[11],q[18];
cx q[2],q[13];
cx q[1],q[2];
rx(3*pi/4) q[1];
cx q[1],q[2];
cx q[2],q[13];
cx q[10],q[21];
cx q[9],q[10];
rx(pi/4) q[9];
cx q[9],q[10];
cx q[10],q[21];
cx q[9],q[10];
cx q[8],q[9];
rx(5*pi/4) q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[16];
rx(5*pi/4) q[10];
cx q[10],q[16];
rx(3*pi/4) q[8];
rx(7*pi/4) q[15];
rx(5*pi/4) q[13];
rx(pi/2) q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(5*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[9],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[15],q[18];
cx q[18],q[11];
rz(pi/4) q[11];
cx q[18],q[11];
cx q[15],q[18];
rz(3*pi/4) q[18];
cx q[4],q[3];
rz(pi/2) q[3];
cx q[4],q[3];
cx q[2],q[12];
cx q[1],q[2];
rx(pi/4) q[1];
cx q[1],q[2];
cx q[2],q[12];
cx q[9],q[10];
rx(7*pi/4) q[9];
cx q[9],q[10];
cx q[1],q[11];
rx(3*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[8];
cx q[1],q[2];
rx(5*pi/4) q[1];
cx q[1],q[2];
cx q[1],q[8];
rx(pi/4) q[15];
cx q[6],q[13];
rx(3*pi/4) q[6];
cx q[6],q[13];
cx q[11],q[14];
rx(3*pi/4) q[11];
cx q[11],q[14];
rz(pi/2) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[18],q[15];
cx q[7],q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[6],q[13];
rx(7*pi/4) q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[21],q[22];
cx q[24],q[15];
cx q[0],q[9];
cx q[3],q[4];
cx q[15],q[24];
cx q[2],q[1];
cx q[6],q[13];
cx q[21],q[18];
cx q[10],q[9];
cx q[9],q[10];
cx q[12],q[17];
cx q[1],q[2];
cx q[16],q[13];
cx q[24],q[15];
cx q[18],q[11];
