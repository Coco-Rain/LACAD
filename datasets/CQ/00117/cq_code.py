import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 30, 15)
.faces(">Z").workplane()
.ellipse(12, 8)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')