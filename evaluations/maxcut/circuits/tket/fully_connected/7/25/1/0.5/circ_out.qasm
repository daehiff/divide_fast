OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[1],q[2];
cx q[10],q[14];
cx q[12],q[16];
cx q[18],q[19];
cx q[1],q[3];
cx q[14],q[16];
cx q[0],q[1];
cx q[0],q[9];
cx q[4],q[1];
cx q[0],q[10];
cx q[1],q[4];
cx q[0],q[13];
cx q[1],q[6];
cx q[4],q[5];
cx q[0],q[17];
cx q[1],q[8];
cx q[6],q[3];
cx q[0],q[18];
cx q[1],q[10];
cx q[8],q[2];
cx q[3],q[6];
cx q[0],q[21];
cx q[1],q[18];
cx q[2],q[8];
cx q[3],q[7];
cx q[0],q[23];
cx q[1],q[20];
cx q[2],q[10];
cx q[5],q[7];
cx q[2],q[11];
cx q[3],q[10];
cx q[5],q[9];
cx q[2],q[12];
cx q[3],q[11];
cx q[4],q[10];
cx q[2],q[13];
cx q[3],q[12];
cx q[6],q[10];
cx q[2],q[17];
cx q[3],q[15];
cx q[5],q[12];
cx q[6],q[11];
cx q[10],q[7];
cx q[2],q[23];
cx q[3],q[20];
cx q[4],q[15];
cx q[5],q[13];
cx q[7],q[10];
cx q[2],q[24];
cx q[3],q[21];
cx q[4],q[18];
cx q[6],q[13];
cx q[7],q[14];
cx q[8],q[10];
cx q[3],q[22];
cx q[4],q[20];
cx q[5],q[18];
cx q[6],q[15];
cx q[8],q[11];
cx q[9],q[14];
cx q[3],q[23];
cx q[4],q[22];
cx q[5],q[20];
cx q[6],q[17];
cx q[8],q[15];
cx q[11],q[13];
cx q[4],q[24];
cx q[5],q[21];
cx q[6],q[19];
cx q[8],q[17];
cx q[11],q[14];
cx q[15],q[13];
cx q[5],q[22];
cx q[6],q[20];
cx q[8],q[19];
cx q[9],q[17];
cx q[14],q[12];
cx q[13],q[15];
cx q[5],q[23];
cx q[7],q[20];
cx q[9],q[18];
cx q[10],q[17];
cx q[12],q[14];
cx q[13],q[16];
cx q[5],q[24];
cx q[6],q[23];
cx q[7],q[21];
cx q[8],q[20];
cx q[10],q[18];
cx q[11],q[17];
cx q[14],q[15];
cx q[6],q[24];
cx q[7],q[22];
cx q[9],q[20];
cx q[10],q[19];
cx q[11],q[18];
cx q[14],q[17];
cx q[15],q[16];
cx q[7],q[23];
cx q[8],q[22];
cx q[9],q[21];
cx q[10],q[20];
cx q[11],q[19];
cx q[13],q[18];
cx q[15],q[17];
cx q[9],q[22];
cx q[10],q[21];
cx q[15],q[18];
cx q[16],q[17];
cx q[9],q[23];
cx q[10],q[22];
cx q[11],q[21];
cx q[16],q[18];
cx q[9],q[24];
cx q[12],q[21];
cx q[13],q[22];
cx q[16],q[20];
cx q[17],q[18];
cx q[10],q[24];
cx q[13],q[23];
cx q[14],q[21];
cx q[16],q[22];
cx q[17],q[19];
cx q[10],q[6];
cx q[12],q[24];
cx q[14],q[23];
cx q[15],q[21];
cx q[18],q[19];
cx q[15],q[23];
cx q[17],q[21];
cx q[18],q[20];
cx q[15],q[24];
cx q[16],q[23];
cx q[17],q[22];
cx q[18],q[21];
cx q[17],q[24];
cx q[18],q[22];
cx q[18],q[24];
cx q[19],q[22];
cx q[19],q[23];
cx q[21],q[22];
cx q[20],q[23];
cx q[20],q[24];
cx q[21],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[22],q[24];
cx q[24],q[23];
cx q[24],q[20];
cx q[24],q[18];
cx q[24],q[17];
cx q[24],q[14];
cx q[24],q[12];
cx q[24],q[9];
cx q[24],q[8];
cx q[24],q[2];
cx q[23],q[8];
cx q[23],q[0];
cx q[24],q[1];
cx q[13],q[1];
cx q[23],q[2];
rz(pi/2) q[24];
cx q[23],q[9];
cx q[23],q[10];
cx q[23],q[15];
cx q[23],q[16];
cx q[23],q[19];
cx q[23],q[20];
cx q[20],q[3];
cx q[23],q[21];
cx q[20],q[4];
cx q[23],q[22];
cx q[22],q[5];
rz(pi/2) q[23];
cx q[22],q[9];
cx q[22],q[10];
cx q[22],q[12];
cx q[22],q[13];
cx q[22],q[15];
cx q[22],q[16];
cx q[22],q[17];
cx q[22],q[18];
cx q[22],q[19];
cx q[22],q[21];
cx q[21],q[0];
rz(pi/2) q[22];
cx q[19],q[0];
cx q[21],q[2];
cx q[19],q[3];
cx q[21],q[5];
cx q[19],q[4];
cx q[20],q[5];
cx q[21],q[9];
cx q[19],q[5];
cx q[20],q[9];
cx q[21],q[10];
cx q[19],q[9];
cx q[20],q[10];
cx q[21],q[11];
cx q[19],q[12];
cx q[21],q[14];
cx q[20],q[16];
cx q[19],q[15];
cx q[21],q[17];
cx q[19],q[16];
cx q[21],q[18];
cx q[20],q[18];
cx q[18],q[0];
cx q[18],q[2];
cx q[18],q[4];
cx q[18],q[5];
cx q[18],q[7];
cx q[18],q[9];
cx q[18],q[11];
cx q[13],q[11];
cx q[18],q[12];
cx q[18],q[15];
cx q[18],q[16];
cx q[18],q[17];
cx q[17],q[0];
cx q[17],q[1];
cx q[17],q[2];
cx q[17],q[5];
cx q[17],q[8];
cx q[17],q[9];
cx q[17],q[12];
cx q[17],q[13];
cx q[17],q[16];
cx q[16],q[1];
rz(pi/2) q[17];
cx q[16],q[2];
cx q[16],q[3];
cx q[16],q[8];
cx q[16],q[9];
cx q[16],q[10];
cx q[16],q[13];
cx q[16],q[14];
cx q[14],q[0];
cx q[16],q[15];
cx q[15],q[1];
cx q[14],q[1];
cx q[15],q[3];
cx q[14],q[4];
cx q[15],q[6];
cx q[14],q[5];
rz(pi/2) q[15];
cx q[11],q[5];
cx q[14],q[6];
cx q[14],q[8];
cx q[14],q[9];
cx q[14],q[13];
cx q[13],q[0];
rz(pi/2) q[14];
cx q[10],q[0];
cx q[13],q[1];
cx q[9],q[0];
cx q[13],q[2];
cx q[10],q[2];
cx q[13],q[7];
cx q[9],q[2];
cx q[10],q[4];
cx q[13],q[8];
cx q[10],q[5];
cx q[13],q[12];
cx q[12],q[3];
cx q[9],q[5];
rz(pi/2) q[13];
cx q[3],q[1];
cx q[12],q[6];
rz(pi/2) q[9];
cx q[11],q[6];
cx q[12],q[7];
cx q[11],q[7];
cx q[12],q[8];
cx q[10],q[7];
cx q[11],q[8];
cx q[10],q[8];
rz(pi/2) q[11];
cx q[8],q[2];
rz(pi/2) q[10];
cx q[8],q[3];
cx q[8],q[4];
cx q[8],q[5];
cx q[8],q[7];
cx q[7],q[2];
rz(pi/2) q[8];
cx q[7],q[5];
cx q[5],q[1];
cx q[7],q[6];
cx q[6],q[2];
rz(pi/2) q[7];
cx q[5],q[2];
cx q[6],q[3];
cx q[3],q[0];
cx q[6],q[4];
cx q[2],q[0];
rz(pi/2) q[3];
cx q[2],q[1];
cx q[1],q[0];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[2],q[0];
cx q[3],q[1];
cx q[3],q[0];
cx q[4],q[3];
cx q[4],q[2];
cx q[5],q[4];
cx q[5],q[3];
cx q[5],q[2];
cx q[5],q[1];
cx q[6],q[5];
cx q[6],q[4];
cx q[6],q[2];
cx q[7],q[4];
cx q[6],q[0];
cx q[7],q[0];
cx q[8],q[6];
cx q[8],q[4];
cx q[9],q[6];
cx q[8],q[1];
cx q[9],q[4];
cx q[10],q[6];
cx q[9],q[2];
cx q[10],q[4];
cx q[10],q[2];
cx q[11],q[9];
cx q[10],q[0];
cx q[11],q[8];
cx q[11],q[5];
cx q[12],q[10];
cx q[11],q[2];
cx q[12],q[8];
cx q[11],q[3];
cx q[12],q[6];
cx q[12],q[1];
cx q[12],q[0];
cx q[13],q[12];
cx q[13],q[10];
cx q[13],q[6];
cx q[14],q[13];
cx q[14],q[11];
cx q[14],q[7];
cx q[14],q[6];
cx q[14],q[4];
cx q[14],q[3];
cx q[14],q[0];
cx q[15],q[14];
cx q[15],q[12];
cx q[18],q[14];
cx q[15],q[11];
cx q[15],q[3];
cx q[15],q[0];
cx q[16],q[15];
cx q[15],q[2];
cx q[16],q[12];
cx q[15],q[9];
cx q[16],q[11];
cx q[18],q[9];
cx q[16],q[10];
cx q[18],q[6];
cx q[16],q[8];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[3];
cx q[16],q[1];
cx q[18],q[3];
cx q[16],q[0];
cx q[19],q[18];
cx q[17],q[16];
cx q[17],q[13];
cx q[17],q[12];
cx q[17],q[11];
cx q[17],q[10];
cx q[17],q[8];
cx q[17],q[4];
cx q[17],q[0];
cx q[19],q[17];
cx q[19],q[16];
cx q[19],q[14];
cx q[19],q[12];
cx q[19],q[9];
cx q[19],q[8];
cx q[19],q[7];
cx q[19],q[4];
cx q[20],q[19];
cx q[20],q[18];
cx q[20],q[17];
cx q[22],q[18];
cx q[20],q[15];
cx q[21],q[17];
cx q[20],q[14];
cx q[21],q[16];
cx q[22],q[17];
cx q[20],q[12];
cx q[21],q[15];
rz(pi/2) q[17];
cx q[20],q[11];
cx q[22],q[12];
cx q[21],q[14];
cx q[20],q[9];
cx q[21],q[11];
cx q[20],q[8];
cx q[22],q[11];
cx q[20],q[7];
cx q[20],q[6];
cx q[21],q[7];
cx q[20],q[4];
cx q[21],q[6];
cx q[20],q[0];
cx q[21],q[5];
cx q[22],q[6];
cx q[21],q[4];
cx q[22],q[5];
rz(pi/2) q[20];
cx q[21],q[2];
cx q[22],q[3];
rz(pi/2) q[4];
rz(pi/2) q[5];
rz(pi/2) q[2];
cx q[21],q[13];
cx q[23],q[22];
cx q[22],q[1];
rz(pi/2) q[13];
cx q[23],q[19];
rz(pi/2) q[21];
cx q[24],q[1];
cx q[19],q[10];
cx q[21],q[13];
cx q[23],q[18];
rz(pi/2) q[22];
rz(pi/2) q[1];
cx q[21],q[2];
cx q[23],q[15];
cx q[21],q[4];
cx q[23],q[14];
cx q[23],q[7];
cx q[23],q[6];
rz(pi/2) q[7];
cx q[23],q[3];
rz(pi/2) q[6];
cx q[23],q[0];
cx q[24],q[3];
rz(pi/2) q[0];
rz(pi/2) q[3];
cx q[24],q[8];
rz(pi/2) q[8];
cx q[24],q[9];
rz(pi/2) q[9];
cx q[24],q[10];
rz(pi/2) q[10];
cx q[24],q[11];
rz(pi/2) q[11];
cx q[24],q[12];
rz(pi/2) q[12];
cx q[24],q[14];
rz(pi/2) q[14];
cx q[24],q[15];
rz(pi/2) q[15];
cx q[24],q[16];
rz(pi/2) q[16];
cx q[24],q[18];
rz(pi/2) q[18];
cx q[24],q[19];
rz(pi/2) q[19];
cx q[24],q[23];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[24],q[23];
cx q[23],q[0];
cx q[24],q[19];
cx q[20],q[0];
cx q[24],q[18];
cx q[20],q[4];
cx q[24],q[16];
cx q[24],q[15];
cx q[24],q[14];
cx q[24],q[12];
cx q[24],q[11];
cx q[24],q[10];
cx q[24],q[9];
cx q[19],q[10];
cx q[24],q[8];
cx q[24],q[3];
cx q[24],q[1];
cx q[23],q[3];
cx q[22],q[1];
cx q[23],q[6];
rz(pi/2) q[24];
cx q[23],q[7];
rx(pi/2) q[24];
cx q[23],q[14];
rz(pi/2) q[24];
cx q[23],q[15];
rz(pi/2) q[24];
cx q[23],q[18];
cx q[23],q[19];
cx q[23],q[22];
cx q[22],q[3];
rz(pi/2) q[23];
cx q[22],q[5];
rx(pi/2) q[23];
cx q[21],q[5];
cx q[22],q[6];
rz(pi/2) q[23];
cx q[21],q[6];
cx q[22],q[11];
rz(pi/2) q[23];
cx q[20],q[6];
cx q[21],q[7];
cx q[22],q[12];
cx q[20],q[7];
cx q[21],q[11];
cx q[22],q[17];
cx q[20],q[8];
cx q[21],q[14];
cx q[22],q[18];
cx q[20],q[9];
cx q[21],q[15];
rz(pi/2) q[22];
cx q[20],q[11];
cx q[21],q[16];
rx(pi/2) q[22];
cx q[20],q[12];
cx q[21],q[17];
rz(pi/2) q[22];
cx q[20],q[14];
rz(pi/2) q[21];
rz(pi/2) q[22];
cx q[20],q[15];
rx(pi/2) q[21];
cx q[20],q[17];
rz(pi/2) q[21];
cx q[20],q[18];
cx q[20],q[19];
cx q[19],q[4];
rz(pi/2) q[20];
cx q[19],q[7];
rx(pi/2) q[20];
cx q[19],q[8];
rz(pi/2) q[20];
cx q[19],q[9];
rz(pi/2) q[20];
cx q[19],q[12];
cx q[19],q[14];
cx q[19],q[16];
cx q[19],q[17];
cx q[17],q[0];
cx q[19],q[18];
cx q[18],q[3];
cx q[17],q[4];
rz(pi/2) q[19];
cx q[18],q[6];
cx q[17],q[8];
rx(pi/2) q[19];
cx q[18],q[9];
cx q[17],q[10];
rz(pi/2) q[19];
cx q[15],q[9];
cx q[17],q[11];
cx q[18],q[14];
cx q[15],q[2];
cx q[17],q[12];
rz(pi/2) q[18];
cx q[17],q[13];
rx(pi/2) q[18];
cx q[17],q[16];
rz(pi/2) q[18];
cx q[16],q[0];
rz(pi/2) q[17];
rz(pi/2) q[18];
cx q[16],q[1];
rx(pi/2) q[17];
cx q[16],q[3];
rz(pi/2) q[17];
cx q[16],q[5];
rz(pi/2) q[17];
cx q[16],q[7];
cx q[16],q[8];
cx q[16],q[10];
cx q[16],q[11];
cx q[16],q[12];
cx q[16],q[15];
cx q[15],q[0];
rz(pi/2) q[16];
cx q[15],q[3];
rx(pi/2) q[16];
cx q[15],q[11];
rz(pi/2) q[16];
cx q[15],q[12];
rz(pi/2) q[16];
cx q[15],q[14];
cx q[14],q[0];
rz(pi/2) q[15];
cx q[14],q[3];
rx(pi/2) q[15];
cx q[14],q[4];
rz(pi/2) q[15];
cx q[14],q[6];
cx q[14],q[7];
cx q[14],q[11];
cx q[11],q[3];
cx q[14],q[13];
cx q[11],q[2];
cx q[13],q[6];
rz(pi/2) q[14];
cx q[11],q[5];
cx q[13],q[10];
rx(pi/2) q[14];
cx q[13],q[12];
rz(pi/2) q[14];
cx q[12],q[0];
rz(pi/2) q[13];
rz(pi/2) q[14];
cx q[12],q[1];
rx(pi/2) q[13];
cx q[12],q[6];
rz(pi/2) q[13];
cx q[12],q[8];
rz(pi/2) q[13];
cx q[11],q[8];
cx q[12],q[10];
cx q[10],q[0];
cx q[8],q[1];
cx q[11],q[9];
rz(pi/2) q[12];
cx q[7],q[0];
cx q[10],q[2];
rz(pi/2) q[11];
rx(pi/2) q[12];
cx q[9],q[2];
cx q[10],q[4];
rx(pi/2) q[11];
rz(pi/2) q[12];
cx q[9],q[4];
cx q[10],q[6];
rz(pi/2) q[11];
rz(pi/2) q[12];
cx q[8],q[4];
cx q[9],q[6];
rz(pi/2) q[10];
cx q[7],q[4];
cx q[8],q[6];
rz(pi/2) q[9];
rx(pi/2) q[10];
cx q[6],q[0];
rz(pi/2) q[7];
rz(pi/2) q[8];
rx(pi/2) q[9];
rz(pi/2) q[10];
cx q[6],q[2];
rx(pi/2) q[7];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[10];
cx q[6],q[4];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(pi/2) q[9];
cx q[6],q[5];
rz(pi/2) q[7];
rz(pi/2) q[8];
cx q[5],q[1];
rz(pi/2) q[6];
cx q[5],q[2];
rx(pi/2) q[6];
cx q[5],q[3];
rz(pi/2) q[6];
cx q[5],q[4];
cx q[4],q[2];
rz(pi/2) q[5];
cx q[4],q[3];
rx(pi/2) q[5];
cx q[3],q[0];
rz(pi/2) q[4];
rz(pi/2) q[5];
cx q[2],q[0];
cx q[3],q[1];
rx(pi/2) q[4];
rz(pi/2) q[0];
cx q[2],q[1];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[1];
rz(pi/2) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[3];
cx q[4],q[0];
cx q[5],q[2];
cx q[6],q[3];
cx q[7],q[6];
cx q[7],q[2];
cx q[7],q[1];
cx q[8],q[7];
cx q[8],q[4];
cx q[8],q[3];
cx q[8],q[2];
cx q[9],q[8];
cx q[9],q[6];
cx q[11],q[8];
cx q[9],q[3];
cx q[11],q[7];
cx q[13],q[8];
cx q[9],q[2];
cx q[10],q[3];
cx q[11],q[4];
cx q[13],q[5];
cx q[9],q[0];
cx q[11],q[2];
cx q[11],q[0];
cx q[12],q[11];
cx q[12],q[9];
cx q[12],q[6];
cx q[12],q[4];
cx q[12],q[1];
cx q[13],q[4];
cx q[12],q[0];
cx q[13],q[1];
cx q[13],q[0];
cx q[14],q[13];
cx q[14],q[12];
cx q[14],q[9];
cx q[14],q[5];
cx q[14],q[1];
cx q[15],q[14];
cx q[15],q[10];
cx q[15],q[9];
cx q[15],q[7];
cx q[15],q[6];
cx q[16],q[15];
cx q[16],q[10];
cx q[16],q[6];
cx q[16],q[5];
cx q[16],q[4];
cx q[16],q[3];
cx q[17],q[16];
cx q[17],q[14];
cx q[18],q[16];
cx q[17],q[13];
cx q[18],q[15];
cx q[17],q[10];
cx q[18],q[13];
cx q[17],q[9];
cx q[18],q[11];
cx q[17],q[8];
cx q[18],q[10];
cx q[17],q[4];
cx q[18],q[6];
cx q[17],q[1];
cx q[18],q[3];
cx q[17],q[0];
cx q[17],q[2];
cx q[19],q[17];
cx q[19],q[13];
cx q[19],q[11];
cx q[19],q[10];
rz(pi/2) q[11];
cx q[19],q[3];
cx q[20],q[19];
cx q[20],q[18];
cx q[20],q[17];
cx q[17],q[8];
cx q[20],q[12];
cx q[20],q[10];
cx q[17],q[14];
cx q[20],q[5];
cx q[20],q[4];
cx q[20],q[3];
cx q[21],q[20];
cx q[21],q[19];
cx q[21],q[15];
cx q[22],q[19];
cx q[21],q[10];
cx q[22],q[18];
cx q[21],q[5];
rz(pi/2) q[10];
cx q[22],q[15];
cx q[22],q[13];
cx q[23],q[21];
cx q[21],q[0];
cx q[22],q[7];
rz(pi/2) q[13];
cx q[23],q[18];
rz(pi/2) q[0];
cx q[22],q[5];
cx q[18],q[9];
cx q[23],q[12];
rz(pi/2) q[21];
cx q[21],q[0];
cx q[22],q[4];
cx q[23],q[7];
rz(pi/2) q[12];
cx q[22],q[3];
cx q[23],q[6];
cx q[6],q[1];
cx q[22],q[2];
cx q[23],q[5];
cx q[24],q[1];
cx q[23],q[4];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[22];
rz(pi/2) q[1];
cx q[24],q[2];
rz(pi/2) q[23];
rz(pi/2) q[2];
cx q[24],q[3];
rz(pi/2) q[3];
cx q[24],q[4];
rz(pi/2) q[4];
cx q[24],q[7];
rz(pi/2) q[7];
cx q[24],q[8];
rz(pi/2) q[8];
cx q[24],q[9];
rz(pi/2) q[9];
cx q[24],q[14];
rz(pi/2) q[14];
cx q[24],q[15];
rz(pi/2) q[15];
cx q[24],q[16];
rz(pi/2) q[16];
cx q[24],q[17];
rz(pi/2) q[17];
cx q[24],q[18];
rz(pi/2) q[18];
cx q[24],q[19];
rz(pi/2) q[19];
cx q[24],q[20];
rz(pi/2) q[20];
rz(pi/2) q[24];
cx q[24],q[20];
cx q[24],q[19];
cx q[24],q[18];
cx q[24],q[17];
cx q[24],q[16];
cx q[24],q[15];
cx q[24],q[14];
cx q[24],q[9];
cx q[17],q[14];
cx q[24],q[8];
cx q[18],q[9];
cx q[24],q[7];
cx q[17],q[8];
cx q[24],q[4];
cx q[24],q[3];
cx q[23],q[4];
cx q[24],q[2];
cx q[23],q[5];
cx q[24],q[1];
cx q[22],q[2];
cx q[6],q[1];
cx q[22],q[3];
cx q[22],q[4];
cx q[23],q[6];
cx q[22],q[5];
cx q[23],q[7];
cx q[22],q[7];
cx q[23],q[12];
cx q[22],q[13];
cx q[23],q[18];
cx q[22],q[15];
cx q[23],q[21];
cx q[21],q[5];
cx q[22],q[18];
rz(pi) q[23];
cx q[21],q[10];
cx q[22],q[19];
cx q[21],q[15];
cx q[21],q[19];
cx q[21],q[20];
cx q[20],q[3];
rz(pi) q[21];
cx q[20],q[4];
rx(pi) q[21];
cx q[20],q[5];
cx q[20],q[10];
cx q[20],q[12];
cx q[20],q[17];
cx q[20],q[18];
cx q[20],q[19];
cx q[19],q[3];
rz(pi) q[20];
cx q[18],q[3];
cx q[19],q[10];
rx(pi) q[20];
cx q[18],q[6];
cx q[19],q[11];
cx q[18],q[10];
cx q[19],q[13];
cx q[18],q[11];
cx q[19],q[17];
cx q[17],q[2];
cx q[18],q[13];
rz(pi) q[19];
cx q[17],q[0];
cx q[18],q[15];
cx q[17],q[1];
cx q[18],q[16];
cx q[17],q[4];
rz(pi) q[18];
cx q[17],q[8];
cx q[17],q[9];
cx q[17],q[10];
cx q[17],q[13];
cx q[17],q[14];
cx q[17],q[16];
cx q[16],q[3];
cx q[16],q[4];
cx q[16],q[5];
cx q[16],q[6];
cx q[16],q[10];
cx q[16],q[15];
cx q[15],q[6];
rz(pi) q[16];
cx q[15],q[7];
cx q[15],q[9];
cx q[15],q[10];
cx q[10],q[3];
cx q[15],q[14];
cx q[14],q[1];
rx(pi) q[10];
rz(pi) q[15];
cx q[14],q[5];
cx q[14],q[9];
cx q[14],q[12];
cx q[14],q[13];
cx q[13],q[0];
cx q[12],q[0];
cx q[13],q[1];
cx q[12],q[1];
cx q[13],q[4];
cx q[12],q[4];
cx q[13],q[5];
cx q[12],q[6];
cx q[13],q[8];
cx q[12],q[9];
cx q[12],q[11];
cx q[11],q[0];
rz(pi) q[12];
cx q[9],q[0];
cx q[11],q[2];
rx(pi) q[12];
cx q[9],q[2];
cx q[11],q[4];
cx q[9],q[3];
cx q[11],q[7];
cx q[9],q[6];
cx q[11],q[8];
cx q[9],q[8];
rx(pi) q[11];
cx q[8],q[2];
rz(pi) q[9];
cx q[8],q[3];
rx(pi) q[9];
cx q[8],q[4];
cx q[4],q[0];
cx q[8],q[7];
rx(pi) q[0];
cx q[7],q[1];
rz(pi) q[8];
cx q[7],q[2];
cx q[5],q[2];
cx q[7],q[6];
cx q[4],q[2];
cx q[6],q[3];
rz(pi) q[7];
cx q[5],q[3];
rx(pi) q[4];
rx(pi) q[7];
cx q[3],q[2];
rz(pi) q[5];
rz(pi) q[2];
rx(pi) q[2];
