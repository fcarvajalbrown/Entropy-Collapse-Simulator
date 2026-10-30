from core.models import FrameData, Node, Member, Load, STEEL_S275

PLANAR_DOFS = [2, 3, 4]


def build() -> FrameData:
    return FrameData(
        name="Single-span beam",
        nodes=_define_nodes(),
        members=_define_members(),
        loads=_define_loads()
    )


def _define_nodes() -> list[Node]:
    return [
        Node(id=0, x=0.0,  y=0.0, z=0.0, fixed_dofs=[0, 1] + PLANAR_DOFS),
        Node(id=1, x=5.0,  y=0.0, z=0.0, fixed_dofs=PLANAR_DOFS),
        Node(id=2, x=10.0, y=0.0, z=0.0, fixed_dofs=[0, 1] + PLANAR_DOFS),
    ]


def _define_members() -> list[Member]:
    return [
        Member(id=0, node_start=0, node_end=1, material=STEEL_S275),
        Member(id=1, node_start=1, node_end=2, material=STEEL_S275),
    ]


def _define_loads() -> list[Load]:
    return [
        Load(node_id=1, dof=1, magnitude=-50_000.0)
    ]
