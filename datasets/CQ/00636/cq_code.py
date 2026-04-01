import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(20, 10)
.faces(">Z").workplane()
.polygon(6, 15)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')