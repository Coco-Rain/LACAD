import cadquery as cq

base_shape = (
cq.Workplane("XY")
.box(20, 20, 10)
)
cut_shape = (
cq.Workplane("XY")
.workplane(offset=5)
.circle(5)
.extrude(20)
)
result = base_shape.cut(cut_shape)
cq.exporters.export(result, 'GT.stl')