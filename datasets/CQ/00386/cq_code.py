import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 5)
.faces(">Z")
.workplane()
.rect(4, 4, forConstruction=False)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')