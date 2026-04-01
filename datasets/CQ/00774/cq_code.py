import cadquery as cq

result = (
cq.Workplane("XY")
.circle(17.75)
.extrude(6)
.faces("<Z")
.workplane()
.polygon(6, 45.5)
.extrude(4)
.faces(">Z")
.workplane()
.polygon(22, 30.5, forConstruction=True)
.vertices()
.hole(3.4)
)
cq.exporters.export(result, 'GT.stl')