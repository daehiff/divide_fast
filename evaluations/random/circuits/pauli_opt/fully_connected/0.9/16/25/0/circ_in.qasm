OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[22],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[4],q[0];
cx q[7],q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[2],q[0];
rz(pi/4) q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[23],q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[4],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[22],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[22],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(3*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[22],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[18],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[12],q[0];
cx q[6],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[15];
cx q[0],q[2];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[2];
cx q[0],q[15];
cx q[0],q[5];
cx q[0],q[24];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[3];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[3];
cx q[0],q[15];
cx q[0],q[18];
cx q[0],q[14];
cx q[5],q[22];
cx q[5],q[9];
cx q[5],q[19];
cx q[5],q[6];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[11];
cx q[5],q[21];
rx(3*pi/4) q[5];
cx q[5],q[21];
cx q[5],q[11];
cx q[5],q[14];
cx q[5],q[17];
cx q[5],q[10];
cx q[5],q[6];
cx q[5],q[19];
cx q[5],q[9];
cx q[5],q[22];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[18],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(5*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[18],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[2];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[13];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[13];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[2];
cx q[0],q[21];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[12];
cx q[0],q[9];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[5];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[5];
cx q[0],q[14];
cx q[0],q[17];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[9];
cx q[0],q[12];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[18];
cx q[2],q[12];
cx q[2],q[6];
cx q[2],q[14];
rx(3*pi/2) q[2];
cx q[2],q[14];
cx q[2],q[6];
cx q[2],q[12];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[6],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[6],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[4];
cx q[14],q[4];
cx q[24],q[4];
cx q[5],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[6],q[4];
cx q[23],q[4];
rz(7*pi/4) q[4];
cx q[23],q[4];
cx q[6],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[5],q[4];
cx q[24],q[4];
cx q[14],q[4];
cx q[17],q[4];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[19];
cx q[0],q[20];
rx(5*pi/4) q[0];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[24];
cx q[12],q[16];
cx q[12],q[21];
rx(pi) q[12];
cx q[12],q[21];
cx q[12],q[16];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[22],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[4],q[0];
cx q[7],q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[2],q[0];
rz(pi/4) q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[23],q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[4],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[22],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[22],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(3*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[22],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[18],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[12],q[0];
cx q[6],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[15];
cx q[0],q[2];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[2];
cx q[0],q[15];
cx q[0],q[5];
cx q[0],q[24];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[3];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[3];
cx q[0],q[15];
cx q[0],q[18];
cx q[0],q[14];
cx q[5],q[22];
cx q[5],q[9];
cx q[5],q[19];
cx q[5],q[6];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[11];
cx q[5],q[21];
rx(3*pi/4) q[5];
cx q[5],q[21];
cx q[5],q[11];
cx q[5],q[14];
cx q[5],q[17];
cx q[5],q[10];
cx q[5],q[6];
cx q[5],q[19];
cx q[5],q[9];
cx q[5],q[22];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[18],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(5*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[18],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[2];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[13];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[13];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[2];
cx q[0],q[21];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[12];
cx q[0],q[9];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[5];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[5];
cx q[0],q[14];
cx q[0],q[17];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[9];
cx q[0],q[12];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[18];
cx q[2],q[12];
cx q[2],q[6];
cx q[2],q[14];
rx(3*pi/2) q[2];
cx q[2],q[14];
cx q[2],q[6];
cx q[2],q[12];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[6],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[6],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[4];
cx q[14],q[4];
cx q[24],q[4];
cx q[5],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[6],q[4];
cx q[23],q[4];
rz(7*pi/4) q[4];
cx q[23],q[4];
cx q[6],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[5],q[4];
cx q[24],q[4];
cx q[14],q[4];
cx q[17],q[4];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[19];
cx q[0],q[20];
rx(5*pi/4) q[0];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[24];
cx q[12],q[16];
cx q[12],q[21];
rx(pi) q[12];
cx q[12],q[21];
cx q[12],q[16];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[22],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[4],q[0];
cx q[7],q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[2],q[0];
rz(pi/4) q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[23],q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[4],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[22],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[22],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(3*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[22],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[18],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[12],q[0];
cx q[6],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[15];
cx q[0],q[2];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[2];
cx q[0],q[15];
cx q[0],q[5];
cx q[0],q[24];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[3];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[3];
cx q[0],q[15];
cx q[0],q[18];
cx q[0],q[14];
cx q[5],q[22];
cx q[5],q[9];
cx q[5],q[19];
cx q[5],q[6];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[11];
cx q[5],q[21];
rx(3*pi/4) q[5];
cx q[5],q[21];
cx q[5],q[11];
cx q[5],q[14];
cx q[5],q[17];
cx q[5],q[10];
cx q[5],q[6];
cx q[5],q[19];
cx q[5],q[9];
cx q[5],q[22];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[18],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(5*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[18],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[2];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[13];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[13];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[2];
cx q[0],q[21];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[12];
cx q[0],q[9];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[5];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[5];
cx q[0],q[14];
cx q[0],q[17];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[9];
cx q[0],q[12];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[18];
cx q[2],q[12];
cx q[2],q[6];
cx q[2],q[14];
rx(3*pi/2) q[2];
cx q[2],q[14];
cx q[2],q[6];
cx q[2],q[12];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[6],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[6],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[4];
cx q[14],q[4];
cx q[24],q[4];
cx q[5],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[6],q[4];
cx q[23],q[4];
rz(7*pi/4) q[4];
cx q[23],q[4];
cx q[6],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[5],q[4];
cx q[24],q[4];
cx q[14],q[4];
cx q[17],q[4];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[19];
cx q[0],q[20];
rx(5*pi/4) q[0];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[24];
cx q[12],q[16];
cx q[12],q[21];
rx(pi) q[12];
cx q[12],q[21];
cx q[12],q[16];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[22],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[4],q[0];
cx q[7],q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[2],q[0];
rz(pi/4) q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[23],q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[4],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[22],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[22],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(3*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[22],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[18],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[12],q[0];
cx q[6],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[15];
cx q[0],q[2];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[2];
cx q[0],q[15];
cx q[0],q[5];
cx q[0],q[24];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[3];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[3];
cx q[0],q[15];
cx q[0],q[18];
cx q[0],q[14];
cx q[5],q[22];
cx q[5],q[9];
cx q[5],q[19];
cx q[5],q[6];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[11];
cx q[5],q[21];
rx(3*pi/4) q[5];
cx q[5],q[21];
cx q[5],q[11];
cx q[5],q[14];
cx q[5],q[17];
cx q[5],q[10];
cx q[5],q[6];
cx q[5],q[19];
cx q[5],q[9];
cx q[5],q[22];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[18],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(5*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[18],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[2];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[13];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[13];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[2];
cx q[0],q[21];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[12];
cx q[0],q[9];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[5];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[5];
cx q[0],q[14];
cx q[0],q[17];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[9];
cx q[0],q[12];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[18];
cx q[2],q[12];
cx q[2],q[6];
cx q[2],q[14];
rx(3*pi/2) q[2];
cx q[2],q[14];
cx q[2],q[6];
cx q[2],q[12];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[6],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[6],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[4];
cx q[14],q[4];
cx q[24],q[4];
cx q[5],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[6],q[4];
cx q[23],q[4];
rz(7*pi/4) q[4];
cx q[23],q[4];
cx q[6],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[5],q[4];
cx q[24],q[4];
cx q[14],q[4];
cx q[17],q[4];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[19];
cx q[0],q[20];
rx(5*pi/4) q[0];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[24];
cx q[12],q[16];
cx q[12],q[21];
rx(pi) q[12];
cx q[12],q[21];
cx q[12],q[16];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[22],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[4],q[0];
cx q[7],q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[2],q[0];
rz(pi/4) q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[23],q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[4],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[22],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[22],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(3*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[22],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[18],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[12],q[0];
cx q[6],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[17],q[0];
cx q[14],q[0];
cx q[24],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[7],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[24],q[0];
cx q[14],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[15];
cx q[0],q[2];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[2];
cx q[0],q[15];
cx q[0],q[5];
cx q[0],q[24];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[3];
cx q[0],q[16];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[16];
cx q[0],q[3];
cx q[0],q[15];
cx q[0],q[18];
cx q[0],q[14];
cx q[5],q[22];
cx q[5],q[9];
cx q[5],q[19];
cx q[5],q[6];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[11];
cx q[5],q[21];
rx(3*pi/4) q[5];
cx q[5],q[21];
cx q[5],q[11];
cx q[5],q[14];
cx q[5],q[17];
cx q[5],q[10];
cx q[5],q[6];
cx q[5],q[19];
cx q[5],q[9];
cx q[5],q[22];
cx q[17],q[0];
cx q[11],q[0];
cx q[14],q[0];
cx q[8],q[0];
cx q[18],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[6],q[0];
cx q[22],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(5*pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[22],q[0];
cx q[6],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[18],q[0];
cx q[8],q[0];
cx q[14],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[2];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[13];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[13];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[2];
cx q[0],q[21];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[12];
cx q[0],q[9];
cx q[0],q[3];
cx q[0],q[22];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[7];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[23];
cx q[0],q[17];
cx q[0],q[14];
cx q[0],q[5];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[5];
cx q[0],q[14];
cx q[0],q[17];
cx q[0],q[23];
cx q[0],q[20];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[22];
cx q[0],q[3];
cx q[0],q[9];
cx q[0],q[12];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[18];
cx q[2],q[12];
cx q[2],q[6];
cx q[2],q[14];
rx(3*pi/2) q[2];
cx q[2],q[14];
cx q[2],q[6];
cx q[2],q[12];
cx q[24],q[0];
cx q[18],q[0];
cx q[21],q[0];
cx q[6],q[0];
cx q[13],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[23],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[23],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[13],q[0];
cx q[6],q[0];
cx q[21],q[0];
cx q[18],q[0];
cx q[24],q[0];
cx q[17],q[4];
cx q[14],q[4];
cx q[24],q[4];
cx q[5],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[6],q[4];
cx q[23],q[4];
rz(7*pi/4) q[4];
cx q[23],q[4];
cx q[6],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[5],q[4];
cx q[24],q[4];
cx q[14],q[4];
cx q[17],q[4];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[12];
cx q[0],q[22];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[16];
cx q[0],q[10];
cx q[0],q[4];
cx q[0],q[23];
cx q[0],q[7];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[7];
cx q[0],q[23];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[16];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[22];
cx q[0],q[12];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[18];
cx q[0],q[24];
cx q[0],q[11];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[5];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[19];
cx q[0],q[20];
rx(5*pi/4) q[0];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[5];
cx q[0],q[24];
cx q[12],q[16];
cx q[12],q[21];
rx(pi) q[12];
cx q[12],q[21];
cx q[12],q[16];
