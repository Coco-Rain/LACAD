import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.rect(5, 5, forConstruction=True)
.vertices()
.hole(2)
)
cq.exporters.export(result, 'GT.stl')