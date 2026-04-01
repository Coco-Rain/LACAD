import cadquery as cq

base_shape = (
cq.Workplane("XY")
.box(30, 30, 15)
)
cut_shape = (
cq.Workplane("XZ")
.workplane(offset=5)
.rect(10, 5)
.extrude(15)
)
result = base_shape.cut(cut_shape)
cq.exporters.export(result, 'GT.stl')