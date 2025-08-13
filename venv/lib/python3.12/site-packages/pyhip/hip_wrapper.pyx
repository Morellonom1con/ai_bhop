cimport cython

cdef extern from "../../../src/hip.c":
    DEF TEXT_LEN = 1024
    DEF MAX_DIM = 3
    DEF IPTR64 = 1
    DEF INT_MAX = 2147483647
    DEF MAX_CPT_CHUNKS = INT_MAX
    DEF MAX_CPT_VXNR = INT_MAX
    DEF MAX_BC_CHAR = 81
    DEF HIP_USE_ULONG = 1
    DEF MAX_ZONE_FIELD_LEN = 8 
    DEF MAX_EDGES_ELEM = 12
    DEF MAX_COLORS = 255
    DEF MAX_DRV_ELEM_TYPES = 29
    DEF MAX_ELEM_TYPES = 6
    DEF MAX_VX_FACE =4
    DEF MAX_EDGES_FACE = 4
    DEF MAX_CHILDS_FACE = 8
    DEF MAX_VX_ELEM = 8
    DEF MAX_FACES_ELEM = 6
    DEF MAX_CHILDS_ELEM = 10
    DEF LEN_GRPNAME = 30
    DEF MAX_UNKNOWNS = 256
    DEF LEN_VARNAME = 30

cdef extern from "../../../src/hip.c":
    void c_init "hip_init"()

    ctypedef enum hip_stat_e:
        success, fatal, warning, info, blank


    ctypedef enum bcGeoType_e:
        bnd, match, inter, duplicateInter, cut,
        noBcGeoType, bndAndInter, any

    ctypedef enum spec_bc_e:
        unspecBc, perBc, wallBc, otherBc

    ctypedef enum specialTopo_e:
        noTopo, axiX, axiY, axiZ, noBc, surf

    ctypedef enum grid_type_enum:
        noGr, mb, uns

    ctypedef enum varCat_e:
        noCat, ns, species, rrates, tpf, rans,
        add, mean, fictive, add_tpf, param, other, hip

    ctypedef struct var_s:
        varCat_e cat
        char grp[LEN_GRPNAME]
        char name[LEN_VARNAME]
        int isVec
        int flag

    ctypedef enum varType_e:
        noVar, cons, prim, primT, para, noType 


    ctypedef struct varList_s:
        int mUnknowns 
        int mUnknFlow
        varType_e varType
        var_s var[MAX_UNKNOWNS]
        double freeStreamVar[MAX_UNKNOWNS]

    ctypedef struct bc_struct:
        char text[MAX_BC_CHAR]
        spec_bc_e bc_e
        char type[MAX_BC_CHAR]
        int nr
        int order
        int refCount
        int mark
        bc_struct *PprvBc
        bc_struct *PnxtBc
        bcGeoType_e geoType
        double llBox[MAX_DIM]
        double urBox[MAX_DIM]
        double rllBox[2]
        double rurBox[2] 


    ctypedef struct block_struct:
        int nr
        char name[TEXT_LEN]
        mb_struct *PmbRoot
        int mVertFile[MAX_DIM]
        int mVert[MAX_DIM]
        int skip
        int mVertsBlock
        double *Pcoor
        int mVertsMarked
        double *PdblMark
        int *PintMark
        double *Punknown
        int mElemsBlock
        int mElemsMarked
        int *PelemMark
        int mSubFaces
        double llBox[MAX_DIM]
        double urBox[MAX_DIM]
        double hMin
        double hMax

    ctypedef struct mb_struct:
        int mBlocks
        block_struct *PblockS
        int mDim
        int mUnknowns

        int mVerts
        int mElems

        int mSubFaces
        varList_s varList


    ctypedef struct color_s:
        unsigned int mBnd
        unsigned int color
        unsigned int mark
        unsigned int mVxColl
        unsigned int maxColl

    ctypedef struct uns_s:
        int nr
        char name[TEXT_LEN]
        grid_struct *pGrid

        int validGrid
        int adapted
        specialTopo_e specialTopo
        int mDim
        double llBox[MAX_DIM]
        double urBox[MAX_DIM]
        double llBoxCyl[2]
        double urBoxCyl[2]
        double hMin
        double hMax
        double volElemMin
        double volElemMax
        double volDomain
        double epsOverlap
        double epsOverlapSq
        color_s *pVxColor
        varList_s varList
        int mBc
        int mBcBnd
        bc_struct **ppBc
        uns_s *pUnsFine
        uns_s *pUnsCoarse
        uns_s *pUnsFinest

    ctypedef struct uns_c:
        grid_struct *PnxtGrid
        grid_struct *PprvGrid
        int nr
        int mDim
        char text[TEXT_LEN]
        varList_s *pVarList
        grid_type_enum type
        uns_s *pUns


    ctypedef struct mb_c:
        grid_struct *PnxtGrid
        grid_struct *PprvGrid
        int nr
        int mDim
        char text[TEXT_LEN]
        varList_s *pVarList
        grid_type_enum type
        mb_struct *Pmb

    ctypedef union grid_struct:
        mb_c mb
        uns_c uns

    ctypedef struct ret_s:
        grid_struct *pGrid
        uns_s *pUns
        hip_stat_e status
        char *msg

    ret_s _call_hip_menus "hip_menu"( char *Pstring )

cpdef hip_init():
    c_init()

cpdef hip_cmd(input_):
    input_bytes = bytes(input_, 'utf-8')
    ret = _call_hip_menus(input_bytes)

    msg = ret.msg.decode("utf-8")
    status = int(ret.status)
    return status, msg
    ### Example for further improvement
    ### To access a member of a structure 
    ### It is done within the same format
    ### As for python objects, for example
    ### Accessing "name" member of "pUns":
        # name = ret.pUns.name.decode("utf-8")
    ### Other examples for non string
        # varlist = ret.pUns.varList
        # nr = int(ret.pUns.nr)
        # mDim = int(ret.pUns.mDim)
        # print(type(varlist))


